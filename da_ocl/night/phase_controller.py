"""
DA-OCL 夜晚流程控制器
编排夜间的蒸馏、梦境生成、梯度更新
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from da_ocl.core.constants import (
    COMPOSITE_DIM,
    DISTILL_LR,
    DREAM_VARIANCE_INIT,
    ENTITY_DIM,
    EWC_FISHER_SAMPLES,
    EWC_LAMBDA,
    GNN_HIDDEN_DIM,
    GRADIENT_PROJECTION_EPS,
    NIGHT_EPOCHS,
    SLOW_MEMORY_DIM,
    SSM_STATE_DIM,
)
from da_ocl.memory.mask_history import MaskHistory
from da_ocl.memory.slow_memory import SlowMemory
from da_ocl.memory.fast_memory import FastMemory
from da_ocl.night.distillation import DistillationPath
from da_ocl.night.dream_generator import DreamGenerator
from da_ocl.night.gradient_projector import GradientProjector
from da_ocl.night.ewc_updater import EWCUpdater
from da_ocl.night.consistency_gate import ConsistencyGate
from da_ocl.losses.night_losses import (
    DistillationLoss,
    KLLoss,
    EWCLoss,
    DreamLoss,
    compute_total_night_loss,
)
from da_ocl.regulation.axes import RegulationMixer


class NightPhaseController(nn.Module):
    """
    夜晚流程控制器。

    编排整个夜间学习流程：
    1. 准备阶段：从 Experience 提取复合特征，生成梦境样本
    2. 优化阶段：蒸馏、快慢对齐、EWC 正则化
    3. 验证阶段：一致性门控，失败则回滚

    梯度更新策略（重要）：
    - 使用 torch.autograd.grad() 计算梯度（而非 optimizer.step()）
    - 使用 GradientProjector.apply_projected_gradients() 手动更新参数
    - 这确保了梯度计算和参数更新完全解耦，不会因 optimizer 状态产生冲突
    """

    def __init__(
        self,
        slow_memory: SlowMemory,
        fast_memory: FastMemory,
        mask_history: MaskHistory,
    ) -> None:
        super().__init__()
        self.slow_memory = slow_memory
        self.fast_memory = fast_memory
        self.mask_history = mask_history

        # 子模块
        self.distillation_path = DistillationPath()
        self.dream_generator = DreamGenerator()
        self.gradient_projector = GradientProjector()
        self.ewc_updater = EWCUpdater()
        self.consistency_gate = ConsistencyGate()

        # 损失函数
        self.distillation_loss_fn = DistillationLoss()
        self.kl_loss_fn = KLLoss()
        self.ewc_loss_fn = EWCLoss()
        self.dream_loss_fn = DreamLoss()

        # 调节器
        self.regulation = RegulationMixer()

    def _prepare_phase(
        self,
        buffer_experiences: list[Any],
        chain_experiences: list[dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        准备阶段：生成梦境样本。
        """
        boundary_centers = None
        if buffer_experiences:
            embeddings = []
            for exp in buffer_experiences:
                if hasattr(exp, "composite") and exp.composite.shape[-1] >= SSM_STATE_DIM:
                    embeddings.append(exp.composite[:SSM_STATE_DIM].detach())
            if embeddings:
                boundary_centers = torch.stack(embeddings)

        return self.dream_generator.generate(
            slow_memory=self.slow_memory,
            chain_experiences=chain_experiences,
            boundary_centers=boundary_centers,
        )

    def _optimization_phase(
        self,
        experiences: list[Any],
        dream_samples: torch.Tensor,
    ) -> dict[str, Any]:
        """
        单轮优化：在一个 epoch 内完成所有经验的损失计算和梯度更新。

        使用 torch.autograd.grad() 而非 optimizer.step() 来计算和手动应用梯度，
        这样避免了跨 epoch 的 optimizer 状态冲突，同时确保慢记忆参数始终参与梯度图。
        """
        if not experiences:
            return {
                "L_distill": 0.0, "L_KL": 0.0,
                "L_EWC": 0.0, "L_dream": 0.0, "L_total": 0.0,
            }

        # 设置新鲜度
        accumulated = self.mask_history.get_accumulated()
        if accumulated:
            float_values = list(accumulated.values())
            freshness = torch.tensor(float_values).mean().unsqueeze(0)
        else:
            freshness = torch.ones(1)
        self.regulation.set_freshness(
            freshness.to(next(self.slow_memory.parameters()).device)
        )

        total_distill = 0.0
        total_kl = 0.0
        total_ewc = 0.0
        total_dream = 0.0
        n = 0

        for exp in experiences:
            if not hasattr(exp, "composite"):
                continue
            if exp.composite.shape[-1] != COMPOSITE_DIM:
                continue

            composite = exp.composite.detach().clone()
            if composite.dim() == 0:
                composite = composite.unsqueeze(0)

            # 提取标量指标（浅提取，不含梯度）
            importance = float(getattr(exp, "importance", 0.5))
            surprise = float(getattr(exp, "surprise", 0.0))

            # ---- 路径A：通过慢记忆建立计算图 ----
            entity_window = composite[:SSM_STATE_DIM]          # (512,) 近似 entity_dim
            if entity_window.dim() == 1:
                entity_window = entity_window.unsqueeze(0)     # (1, 512)
            if entity_window.shape[-1] != ENTITY_DIM:
                entity_window = entity_window[..., :ENTITY_DIM]
                if entity_window.dim() == 1:
                    entity_window = entity_window.unsqueeze(0)

            # 完整前向传播，建立慢记忆的计算图
            gnn_out = self.slow_memory.gnn(entity_window)
            O_batch = self.slow_memory.ssm(gnn_out).squeeze(0)  # (SSM_STATE_DIM,)

            # 快记忆检索（detach 结果，避免干扰梯度图）
            try:
                fast_retrieved_full, _ = self.fast_memory.retrieve(composite)
                fast_retrieved = fast_retrieved_full[:SSM_STATE_DIM].detach()
            except RuntimeError:
                device = entity_window.device
                fast_retrieved = torch.zeros(SSM_STATE_DIM, device=device, dtype=O_batch.dtype)

            # 蒸馏损失（建立在计算图上）
            distill_loss = self.distillation_loss_fn(
                O_batch, fast_retrieved, importance, surprise
            )

            # KL 损失（建立在计算图上）
            slow_probs = F.softmax(O_batch, dim=-1).clamp(min=1e-8)
            fast_probs = F.softmax(fast_retrieved, dim=-1).clamp(min=1e-8)
            kl_loss = self.kl_loss_fn(slow_probs, fast_probs, importance, surprise)

            # EWC 损失（建立在慢记忆参数上的计算图）
            ewc_loss = self.ewc_loss_fn(self.slow_memory)

            # ---- 梯度计算（单次 autograd.grad，避免图冲突）----
            # 所有损失都建立在慢记忆参数的计算图上，可一次性获取梯度。
            # EWC 损失的图：param - old_param → (diff^2) * fisher → sum
            # 前向损失的图：slow_memory(entity_window) → O_batch → distill/KL loss
            params = [p for p in self.slow_memory.parameters() if p.requires_grad]
            if params:
                # allow_unused=True：某些参数（如 rules 无关层）可能不参与特定损失
                grads = torch.autograd.grad(
                    outputs=[distill_loss, kl_loss, ewc_loss],
                    inputs=params,
                    retain_graph=False,
                    allow_unused=True,
                )
            else:
                grads = []

            # 构建梯度字典（用于投影）
            grad_dict: dict[str, torch.Tensor] = {}
            for (pname, _), g in zip(self.slow_memory.named_parameters(), grads):
                if g is not None:
                    grad_dict[pname] = g

            # ---- 梯度投影 + 手动应用 ----
            projected = self.gradient_projector.project(grad_dict, self.mask_history)
            self.gradient_projector.apply_projected_gradients(
                self.slow_memory, projected, DISTILL_LR
            )

            # 记录损失值
            total_distill += float(distill_loss.detach())
            total_kl += float(kl_loss.detach())
            total_ewc += float(ewc_loss.detach())
            n += 1

        if n == 0:
            return {"L_distill": 0.0, "L_KL": 0.0, "L_EWC": 0.0, "L_dream": 0.0, "L_total": 0.0}

        avg = {
            "L_distill": total_distill / n,
            "L_KL": total_kl / n,
            "L_EWC": total_ewc / n,
            "L_dream": total_dream / n,
            "L_total": (total_distill + total_kl + total_ewc + total_dream) / n,
        }
        return avg

    def _validation_phase(
        self,
        before_state: dict[str, torch.Tensor],
    ) -> tuple[bool, float]:
        """
        验证阶段：一致性门控检查。
        """
        after_state = {n: p.detach() for n, p in self.slow_memory.named_parameters()}
        accepted, score = self.consistency_gate.evaluate(before_state, after_state)
        if not accepted:
            self.consistency_gate.validate_and_rollback(
                self.slow_memory, before_state, after_state
            )
        return accepted, score

    def run(
        self,
        buffer_experiences: list[Any],
        chain_experiences: list[dict] | None = None,
        epochs: int = NIGHT_EPOCHS,
    ) -> dict[str, Any]:
        """
        运行整个夜晚流程。
        """
        before_state = {n: p.detach().clone() for n, p in self.slow_memory.named_parameters()}
        self.ewc_updater.snapshot(self.slow_memory)

        dream_output = self._prepare_phase(buffer_experiences, chain_experiences)
        dream_samples = dream_output["dream_samples"]

        all_components = []
        accepted = True
        consistency_score = 1.0

        for _ in range(epochs):
            components = self._optimization_phase(buffer_experiences, dream_samples)
            all_components.append(components)
            accepted, consistency_score = self._validation_phase(before_state)
            if not accepted:
                break

        avg_losses = {}
        if all_components:
            keys = all_components[0].keys()
            for k in keys:
                avg_losses[k] = sum(c[k] for c in all_components) / len(all_components)

        # Fisher 更新（使用 detach 后的输入）
        if buffer_experiences:
            first = buffer_experiences[0]
            if hasattr(first, "composite") and first.composite.shape[-1] >= SSM_STATE_DIM:
                inp = first.composite[:SSM_STATE_DIM].detach().unsqueeze(0)  # (1, 512)
                try:
                    self.ewc_updater.compute_fisher(self.slow_memory, inp)
                except Exception:
                    pass

        return {
            "accepted": accepted,
            "consistency_score": consistency_score,
            "avg_losses": avg_losses,
            "num_dream_samples": dream_samples.shape[0],
            "epochs_run": len(all_components),
        }

    def reset(self) -> None:
        """重置夜晚状态"""
        self.mask_history.reset()
        self.ewc_updater = EWCUpdater()
