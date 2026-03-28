"""
DA-OCL 白天阶段总控制器
整合白天所有子模块的编排
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from da_ocl.causal.boundary_detector import BoundaryDetector
from da_ocl.causal.coverage_tracker import CoverageTracker
from da_ocl.causal.window_manager import CausalWindowManager
from da_ocl.core.composite_builder import CompositeFeatureBuilder
from da_ocl.core.constants import ENTITY_DIM, WINDOW_SIZE_DEFAULT
from da_ocl.core.encoder import Encoder
from da_ocl.day.adversarial_filter import AdversarialFilter, FilterResult
from da_ocl.day.metrics import compute_importance, compute_surprise, is_death_experience
from da_ocl.memory.fast_memory import FastMemory
from da_ocl.memory.joint_mask import compute_joint_mask
from da_ocl.memory.mask_history import MaskHistory
from da_ocl.memory.slow_memory import SlowMemory


@dataclass
class DayMetrics:
    """白天单步返回的指标"""
    surprise: float
    importance: float
    coverage_rate: float
    mask_value: float
    filter_passed: bool
    filter_result: FilterResult | None
    new_boundary: bool
    added_to_buffer: bool
    is_death: bool
    timestep: int
    entity_window_available: bool


class DayPhaseController:
    """
    白天阶段总控制器。

    整合白天所有子模块，管理推理和训练流程。
    """

    def __init__(
        self,
        encoder: Encoder,
        window_manager: CausalWindowManager,
        slow_memory: SlowMemory,
        fast_memory: FastMemory,
        mask_history: MaskHistory,
        coverage_tracker: CoverageTracker,
        boundary_detector: BoundaryDetector,
        adversarial_filter: AdversarialFilter,
    ) -> None:
        self.encoder = encoder
        self.window_manager = window_manager
        self.slow_memory = slow_memory
        self.fast_memory = fast_memory
        self.mask_history = mask_history
        self.coverage_tracker = coverage_tracker
        self.boundary_detector = boundary_detector
        self.adversarial_filter = adversarial_filter

        self._composite_builder = CompositeFeatureBuilder()
        self._step_count: int = 0

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        推理接口：给定观测，返回动作（或动作分布）。

        当前简化实现：返回随机动作。

        Args:
            obs: (obs_dim,) 单帧观测

        Returns:
            action tensor
        """
        obs = obs.to(next(self.encoder.parameters()).device)
        entity = self.encoder(obs)
        self.window_manager.push(entity)
        return torch.randint(0, 32, (1,))

    def step(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        r_penalty: float = 0.0,
    ) -> DayMetrics:
        """
        白天单步流程。

        Args:
            obs: (obs_dim,) 当前帧观测
            action: 执行的动作索引
            reward: 即时奖励
            done: episode 是否结束
            r_penalty: 死亡惩罚值（由环境 info 提供）

        Returns:
            DayMetrics 字典
        """
        self._step_count += 1
        device = next(self.encoder.parameters()).device
        obs = obs.to(device)
        entity = self.encoder(obs)

        # === 路径B（并行）：快记忆预写入准备 ===
        # 因果窗口累积
        self.window_manager.push(entity)
        entity_window = self.window_manager.get_batch()

        # 窗口未满时：返回预热指标
        if entity_window is None:
            return DayMetrics(
                surprise=0.0,
                importance=0.0,
                coverage_rate=0.0,
                mask_value=0.0,
                filter_passed=False,
                filter_result=None,
                new_boundary=False,
                added_to_buffer=False,
                is_death=is_death_experience(r_penalty),
                timestep=self._step_count,
                entity_window_available=False,
            )

        # === 路径A（并行）：慢记忆检索 ===
        o_hat_batch = self.slow_memory.retrieve(entity_window)

        # === 路径A同步：慢记忆三层前向 + 沉默写入 ===
        _, o_batch, _ = self.slow_memory(entity_window)

        # === 汇合点：复合特征构建 ===
        composite = self._composite_builder.build(o_batch, o_hat_batch)

        # === 惊喜度和覆盖追踪 ===
        surprise_scalar, surprise_per_frame = compute_surprise(o_batch, o_hat_batch)
        coverage_rate = self.coverage_tracker.get_batch_coverage_rate()

        # === 情景边界检测 ===
        new_boundary = self.boundary_detector.detect(surprise_scalar)

        # === 联合掩码计算 + 沉默写入 ===
        mask_value = compute_joint_mask(surprise_per_frame, coverage_rate)
        self.slow_memory.silent_write(entity_window, mask_value)

        # 累积到掩码历史
        self.mask_history.accumulate_from_model(self.slow_memory, mask_value)

        # === 对抗滤波 ===
        filter_result = self.adversarial_filter.filter(composite, surprise_per_frame)

        # === 双指标计算 ===
        importance = compute_importance(coverage_rate, r_penalty)
        is_death = is_death_experience(r_penalty)

        # === 快记忆写入（通过 Filter 的经验）===
        added_to_buffer = False
        if filter_result.passed and not is_death:
            # 通过滤波且非死亡：写入快记忆
            self.fast_memory.store(composite.detach())
            added_to_buffer = True
        elif is_death:
            # 死亡经验：进入保护区（由 BufferManager 处理）
            added_to_buffer = False

        return DayMetrics(
            surprise=surprise_scalar,
            importance=importance,
            coverage_rate=coverage_rate,
            mask_value=mask_value,
            filter_passed=filter_result.passed,
            filter_result=filter_result,
            new_boundary=new_boundary,
            added_to_buffer=added_to_buffer,
            is_death=is_death,
            timestep=self._step_count,
            entity_window_available=True,
        )

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self) -> None:
        """重置白天状态（episode 结束时）"""
        self.window_manager.reset()
        self._step_count = 0
