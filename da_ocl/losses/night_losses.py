"""
DA-OCL 损失函数集合
L_distill, L_KL, L_EWC, L_dream, L_night

规格符合 SPEC.md §6.14
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

from da_ocl.core.constants import (
    ALPHA_KL,
    BETA0_KL,
    COMPOSITE_DIM,
    EWC_LAMBDA,
    GAMMA_DREAM,
    RULE_DIM,
    SSM_STATE_DIM,
    SURPRISE_SIGMA,
    LAMBDA_DECAY,
)


class DistillationLoss(nn.Module):
    """
    蒸馏损失：L_distill。

    L_distill = weight * MSE(slow_output, fast_retrieved)
    weight = importance * (1 + surprise)
    将快记忆的联想检索结果蒸馏到慢记忆。
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        slow_output: torch.Tensor,
        fast_retrieved: torch.Tensor,
        importance: float,
        surprise: float,
    ) -> torch.Tensor:
        """
        Args:
            slow_output: (SSM_STATE_DIM,) 慢记忆输出
            fast_retrieved: (SSM_STATE_DIM,) 快记忆检索结果
            importance: 重要性分数
            surprise: 惊喜度

        Returns:
            蒸馏损失标量
        """
        # 确保维度匹配
        min_dim = min(slow_output.shape[-1], fast_retrieved.shape[-1])
        mse = F.mse_loss(
            slow_output[..., :min_dim],
            fast_retrieved[..., :min_dim],
        )
        # 权重：importance ∈ [0,1], surprise ∈ [0,+∞)
        weight = importance * (1.0 + min(surprise, 10.0))
        return weight * mse


class KLLoss(nn.Module):
    """
    KL 散度损失：L_KL。

    规格：
    beta = beta0 * 1/(1 + alpha * Imp) * (1 - exp(-Surp/sigma)) * exp(-lambda * Delta_t)

    语义：当重要性低时，慢记忆向快记忆对齐（强化已知的模式）；
          当重要性高时，慢记忆保持独立（避免被快速变化覆盖）。
    """

    def __init__(
        self,
        beta0: float = BETA0_KL,
        alpha: float = ALPHA_KL,
        sigma: float = SURPRISE_SIGMA,
        lambda_decay: float = LAMBDA_DECAY,
    ) -> None:
        super().__init__()
        self.beta0 = beta0
        self.alpha = alpha
        self.sigma = sigma
        self.lambda_decay = lambda_decay

    def forward(
        self,
        slow_probs: torch.Tensor,
        fast_probs: torch.Tensor,
        importance: float,
        surprise: float = 0.0,
        delta_t: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            slow_probs: (N,) 慢记忆输出经 softmax 的概率分布
            fast_probs: (N,) 快记忆输出经 softmax 的概率分布
            importance: 重要性分数
            surprise: 惊喜度
            delta_t: 自上次更新以来的时间步数

        Returns:
            KL 散度损失标量
        """
        # beta = beta0 * 1/(1+alpha*Imp) * (1-exp(-Surp/sigma)) * exp(-lambda*Delta_t)
        importance_term = 1.0 / (1.0 + self.alpha * importance)
        surprise_term = 1.0 - torch.exp(torch.tensor(-surprise / max(self.sigma, 1e-8)))
        time_term = torch.exp(torch.tensor(-self.lambda_decay * delta_t))
        beta = self.beta0 * importance_term * surprise_term * time_term

        # 确保维度匹配
        min_dim = min(slow_probs.shape[-1], fast_probs.shape[-1])
        sp = slow_probs[..., :min_dim].clamp(min=1e-8)
        fp = fast_probs[..., :min_dim].clamp(min=1e-8)

        # KL(slow || fast) — 慢记忆向快记忆对齐
        kl = F.kl_div(fp.log(), sp, reduction="batchmean")
        return beta * kl


class EWCLoss(nn.Module):
    """
    弹性权重巩固损失：L_EWC。

    L_EWC = lambda * Σ_i F_i * (theta_i - theta_i_old)^2

    F_i = Fisher 信息矩阵对角线（参数重要性的代理）
    """

    def __init__(self, ewc_lambda: float = EWC_LAMBDA) -> None:
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self._fisher_diagonal: dict[str, torch.Tensor] = {}
        self._old_params: dict[str, torch.Tensor] = {}

    def snapshot(self, model: nn.Module) -> None:
        """记录当前参数快照"""
        self._old_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    def compute_fisher(self, model: nn.Module, data: torch.Tensor) -> None:
        """
        计算 Fisher 信息矩阵对角线。

        Fisher 信息作为参数重要性的代理：在参数梯度方向方差大的参数更重要。

        Args:
            model: 慢记忆模型
            data: (N, D) 用于 Fisher 计算的数据样本，N 个样本，维度 D
        """
        model.eval()
        self._old_params = {n: p.detach().clone() for n, p in model.named_parameters()}

        # 简化 Fisher：对角线近似 = 参数梯度平方的期望
        fisher: dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        num_samples = data.shape[0]
        num_iters = min(num_samples, 10)

        for i in range(num_iters):
            model.zero_grad()
            # 取第 i 个样本（完整维度）
            x = data[i]
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (D,) → (1, D)
            # 使用模型前向传播建立计算图，使参数获得梯度
            output = model(x)
            loss = output.mean()
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= num_iters

        self._fisher_diagonal = fisher
        model.train()

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        计算 EWC 正则化损失。

        Args:
            model: 慢记忆模型（当前参数状态）

        Returns:
            EWC 损失标量
        """
        if not self._old_params:
            # 必须返回 requires_grad=True 的 tensor，backward() 才能成功
            return torch.tensor(0.0, requires_grad=True)

        loss = 0.0
        for n, param in model.named_parameters():
            if n in self._old_params:
                fisher = self._fisher_diagonal.get(n)
                if fisher is not None:
                    diff = (param - self._old_params[n]) ** 2
                    loss += (fisher * diff).sum()

        return self.ewc_lambda * loss


class DreamLoss(nn.Module):
    """
    梦境损失：L_dream。

    L_dream = gamma * H(distribution)

    梦境生成器在规律边界采样，使用快记忆中检索分布的熵作为梦境损失。
    高熵 → 规律边界模糊 → 高损失。
    """

    def __init__(self, gamma: float = GAMMA_DREAM) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        slow_output: torch.Tensor,
        retrieved_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slow_output: (SSM_STATE_DIM,) 慢记忆输出
            retrieved_distribution: (N,) 快记忆检索注意力分布

        Returns:
            梦境损失标量
        """
        # 使用分布熵作为梦境损失（高熵 → 规律边界模糊 → 高损失）
        dist = retrieved_distribution.clamp(min=1e-8)
        entropy = -(dist * dist.log()).sum()
        return self.gamma * entropy


def compute_total_night_loss(
    distillation_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    ewc_loss: torch.Tensor,
    dream_loss: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    合并夜晚总损失。

    L_night = L_distill + L_KL + L_EWC + L_dream

    Args:
        distillation_loss: 蒸馏损失
        kl_loss: KL 散度损失
        ewc_loss: EWC 正则化损失
        dream_loss: 梦境损失

    Returns:
        (total_loss, loss_components)
    """
    total = distillation_loss + kl_loss + ewc_loss + dream_loss

    components = {
        "L_distill": distillation_loss.detach() if hasattr(distillation_loss, "detach") else distillation_loss,
        "L_KL": kl_loss.detach() if hasattr(kl_loss, "detach") else kl_loss,
        "L_EWC": ewc_loss.detach() if hasattr(ewc_loss, "detach") else ewc_loss,
        "L_dream": dream_loss.detach() if hasattr(dream_loss, "detach") else dream_loss,
        "L_total": total.detach() if hasattr(total, "detach") else total,
    }

    return total, components
