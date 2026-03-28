"""
DA-OCL Agent — 完整集成
整合白天流程 + 三区 Buffer + 夜晚流程
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from da_ocl.buffer.three_zone_buffer import ThreeZoneBuffer
from da_ocl.buffer.experience import Experience
from da_ocl.causal.boundary_detector import BoundaryDetector
from da_ocl.causal.window_manager import CausalWindowManager
from da_ocl.causal.surprise_detector import SurpriseDetector
from da_ocl.core.encoder import Encoder
from da_ocl.core.gnn_layer import GNNRelationEncoder
from da_ocl.core.ssm_layer import SSMCompressor
from da_ocl.core.rule_constraints import RuleConstraintLayer
from da_ocl.core.composite_builder import CompositeFeatureBuilder
from da_ocl.day.adversarial_filter import AdversarialFilter
from da_ocl.day.metrics import compute_importance, compute_surprise, is_death_experience
from da_ocl.memory.fast_memory import FastMemory
from da_ocl.memory.slow_memory import SlowMemory
from da_ocl.memory.mask_history import MaskHistory
from da_ocl.memory.joint_mask import JointMaskComputer
from da_ocl.night.night_pipeline import NightPipeline
from da_ocl.night.night_trigger import NightTrigger
from envs.base_env import BaseEnv, EnvOutput
from da_ocl.core.constants import (
    ACTION_DIM,
    COMPOSITE_DIM,
    GNN_HIDDEN_DIM,
    RULE_DIM,
    SSM_STATE_DIM,
    WINDOW_SIZE_DEFAULT,
)


class DAOCLAgent(nn.Module):
    """
    DA-OCL 完整智能体。

    整合所有组件：
    - Encoder + GNN + SSM + RuleConstraint（感知层）
    - FastMemory + SlowMemory（双记忆系统）
    - JointMask + MaskHistory（联合掩码机制）
    - DayPhaseController + BoundaryDetector（白天流程）
    - ThreeZoneBuffer + DoubleBufferManager（经验管理）
    - NightPipeline（夜晚流程）
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 感知层 ----
        self.encoder = Encoder(obs_dim=env.obs_dim)
        self.gnn = GNNRelationEncoder(entity_dim=env.obs_dim)
        self.ssm = SSMCompressor(gnn_hidden_dim=GNN_HIDDEN_DIM, ssm_state_dim=SSM_STATE_DIM)
        self.rule_constraints = RuleConstraintLayer()

        # ---- 双记忆系统 ----
        self.fast_memory = FastMemory()
        self.slow_memory = SlowMemory()

        # ---- 掩码机制 ----
        self.joint_mask_computer = JointMaskComputer()
        self.mask_history = MaskHistory()

        # ---- 白天流程 ----
        self.causal_window = CausalWindowManager(window_size=WINDOW_SIZE_DEFAULT)
        self.surprise_detector = SurpriseDetector()
        self.boundary_detector = BoundaryDetector()
        self.adversarial_filter = AdversarialFilter()

        # ---- Buffer 系统 ----
        self.buffer = ThreeZoneBuffer()

        # ---- 夜晚流程 ----
        self.night_trigger = NightTrigger()
        self.night_pipeline = NightPipeline(
            slow_memory=self.slow_memory,
            fast_memory=self.fast_memory,
            mask_history=self.mask_history,
            buffer=self.buffer,
            night_trigger=self.night_trigger,
        )

        # ---- 状态追踪 ----
        self._timestep: int = 0
        self._is_night: bool = False
        self._cumulated_surprise: list[float] = []

    def reset(self) -> None:
        """重置智能体状态（episode 开始时调用）"""
        self._timestep = 0
        self._is_night = False
        self._cumulated_surprise = []

        # 重置所有子模块
        self.causal_window.reset()
        self.surprise_detector.reset()
        self.boundary_detector.reset()
        self.mask_history.reset()
        self.buffer.reset()

    def observe(self, env_output: EnvOutput) -> dict[str, Any]:
        """
        日间观测处理主入口。

        Args:
            env_output: 环境输出

        Returns:
            处理结果字典
        """
        obs = env_output.obs.to(self.device)

        # 1. 编码
        encoded = self.encoder(obs.unsqueeze(0))  # (1, entity_dim)

        # 2. GNN + SSM 压缩
        gnn_out = self.gnn(encoded)  # (1, gnn_hidden_dim)
        ssm_out = self.ssm(gnn_out)  # (SSM_STATE_DIM,)
        ssm_state = ssm_out.unsqueeze(0)  # (1, SSM_STATE_DIM)

        # 3. 更新因果窗口（存储编码后的实体表征）
        self.causal_window.push(encoded.squeeze(0).detach().cpu())

        # 4. 规则约束
        constrained = self.rule_constraints(ssm_state)

        # 5. 构建复合特征（用于快记忆存储）
        composite = CompositeFeatureBuilder.build(ssm_out, ssm_out)

        # 6. 快记忆存储
        self.fast_memory.store(composite)

        # 7. 快记忆检索（安全处理空记忆）
        try:
            fast_retrieved, _ = self.fast_memory.retrieve(composite)
        except RuntimeError:
            fast_retrieved = torch.zeros(COMPOSITE_DIM, device=ssm_state.device, dtype=ssm_state.dtype)

        # 8. 计算惊喜度（只用前 SSM_STATE_DIM 部分）
        fast_ssm = fast_retrieved[:SSM_STATE_DIM]
        scalar_surprise, per_frame = compute_surprise(ssm_out, fast_ssm)
        self._cumulated_surprise.append(scalar_surprise)
        self.surprise_detector.update(scalar_surprise)
        self.mask_history.record_surprise(scalar_surprise)

        # 8. 计算联合掩码
        joint_mask = self.joint_mask_computer.compute_joint_mask(
            surprise=scalar_surprise,
            coverage_rate=0.5,
        )

        # 9. 掩码历史累积
        for name, _ in self.slow_memory.named_parameters():
            self.mask_history.accumulate(name, joint_mask)

        # 10. 对抗滤波
        filtered = self.adversarial_filter.filter(
            composite=composite,
            surprise_per_frame=per_frame,
        )

        # 11. 边界检测
        avg_surprise = sum(self._cumulated_surprise[-20:]) / len(self._cumulated_surprise[-20:]) if self._cumulated_surprise else 0.0
        is_boundary = self.boundary_detector.detect(avg_surprise)

        # 12. 计算重要性
        importance = compute_importance(
            coverage_rate=0.5,
            r_penalty=env_output.reward,
        )
        death = is_death_experience(env_output.reward)

        exp = Experience(
            entity_window=encoded.squeeze(0),
            composite=composite,
            ssm_output=ssm_state.squeeze(0),
            surprise=scalar_surprise,
            importance=importance,
            coverage_rate=0.5,
            r_penalty=env_output.reward if env_output.reward < 0 else 0.0,
            is_death=death,
            timestep=self._timestep,
            priority_score=importance + scalar_surprise,
            pattern_overlap=0.0,
            task_value=importance,
        )

        self.buffer.add(experience=exp)

        # 14. 检查夜晚触发
        should_night, night_reason = self.night_pipeline.check_trigger(
            timestep=self._timestep,
        )

        return {
            "obs": obs,
            "ssm_state": ssm_state,
            "fast_retrieved": fast_retrieved,
            "surprise": scalar_surprise,
            "joint_mask": joint_mask,
            "is_boundary": is_boundary,
            "composite": composite,
            "should_night": should_night,
            "night_reason": night_reason,
        }

    def run_night_phase(self) -> dict[str, Any]:
        """
        触发夜晚流程。

        Returns:
            夜晚执行结果
        """
        self._is_night = True

        # 运行夜晚管道
        result = self.night_pipeline.run(timestep=self._timestep)

        # 重置日间状态
        self.mask_history.reset()
        self._cumulated_surprise = []

        self._is_night = False
        return result

    def step(self, env_output: EnvOutput) -> dict[str, Any]:
        """
        单步执行（日间观测 + 夜晚触发检查）。

        Args:
            env_output: 环境输出

        Returns:
            处理结果
        """
        result = self.observe(env_output)

        # 检查夜晚触发
        if result["should_night"]:
            night_result = self.run_night_phase()
            result["night_result"] = night_result

        self._timestep += 1
        return result

    def get_statistics(self) -> dict[str, Any]:
        """获取智能体统计信息"""
        return {
            "timestep": self._timestep,
            "is_night": self._is_night,
            "buffer_size": self.buffer.size(),
            "mask_history_summary": self.mask_history.summary(),
            "night_stats": self.night_pipeline.get_statistics(),
        }
