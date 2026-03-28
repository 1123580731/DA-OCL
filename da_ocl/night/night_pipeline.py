"""
DA-OCL 夜晚并行+串行编排器
管理夜间学习流程的完整生命周期
"""

from __future__ import annotations

from typing import Any

import torch

from da_ocl.buffer.three_zone_buffer import ThreeZoneBuffer
from da_ocl.buffer.experience import Experience
from da_ocl.memory.mask_history import MaskHistory
from da_ocl.memory.slow_memory import SlowMemory
from da_ocl.memory.fast_memory import FastMemory
from da_ocl.night.phase_controller import NightPhaseController
from da_ocl.night.night_trigger import NightTrigger
from da_ocl.core.constants import SSM_STATE_DIM


class NightPipeline:
    """
    夜晚流程编排器。

    负责：
    1. 触发检测（并行：监控多条条件）
    2. 并行准备（梦境生成 + Fisher 计算可并行）
    3. 串行优化（梯度更新需要串行）
    4. 结果回写（更新慢记忆、写回 Buffer）
    """

    def __init__(
        self,
        slow_memory: SlowMemory,
        fast_memory: FastMemory,
        mask_history: MaskHistory,
        buffer: ThreeZoneBuffer,
        night_trigger: NightTrigger | None = None,
    ) -> None:
        self.slow_memory = slow_memory
        self.fast_memory = fast_memory
        self.mask_history = mask_history
        self.buffer = buffer
        self.night_trigger = night_trigger or NightTrigger()

        # 夜晚控制器
        self.controller = NightPhaseController(
            slow_memory=slow_memory,
            fast_memory=fast_memory,
            mask_history=mask_history,
        )

        # 夜间统计
        self._night_count: int = 0
        self._total_nights: int = 0
        self._rejected_nights: int = 0

    def check_trigger(
        self,
        timestep: int,
        buffer_window_count: int | None = None,
        mask_density: float | None = None,
        avg_surprise: float | None = None,
    ) -> tuple[bool, str]:
        """
        检查是否应触发夜晚。

        Args:
            timestep: 当前时间步
            buffer_window_count: Buffer 中的窗口数量
            mask_density: 掩码激活密度
            avg_surprise: 平均惊喜度

        Returns:
            (should_trigger, reason)
        """
        # 如果没有提供，自动计算
        if buffer_window_count is None:
            buffer_window_count = self.buffer.size()

        if mask_density is None:
            mask_density = self.mask_history.get_density()

        if avg_surprise is None:
            avg_surprise = self.mask_history.get_avg_surprise()

        return self.night_trigger.should_trigger(
            buffer_window_count=buffer_window_count,
            mask_density=mask_density,
            avg_surprise=avg_surprise,
            timestep=timestep,
        )

    def run(
        self,
        timestep: int,
        chain_experiences: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        运行完整的夜晚流程。

        Args:
            timestep: 当前时间步
            chain_experiences: 因果链经验

        Returns:
            夜晚执行结果
        """
        self._night_count += 1
        self._total_nights += 1

        # 获取 Buffer 中的经验
        buffer_experiences = self.buffer.get_all_experiences()

        if not buffer_experiences:
            return {
                "night_count": self._night_count,
                "status": "skipped",
                "reason": "empty_buffer",
                "accepted": True,
            }

        # 提取因果链经验
        if chain_experiences is None:
            chain_experiences = self._extract_chain_experiences(buffer_experiences)

        # 运行夜晚控制器
        result = self.controller.run(
            buffer_experiences=buffer_experiences,
            chain_experiences=chain_experiences,
        )

        # 统计拒绝
        if not result["accepted"]:
            self._rejected_nights += 1

        # 后处理：更新 Buffer 元数据
        self._post_process_buffer(buffer_experiences, result)

        return {
            "night_count": self._night_count,
            "status": "completed" if result["accepted"] else "rejected",
            "reason": "consistency_gate_failed" if not result["accepted"] else "success",
            "accepted": result["accepted"],
            "consistency_score": result["consistency_score"],
            "avg_losses": result["avg_losses"],
            "num_dream_samples": result["num_dream_samples"],
            "epochs_run": result["epochs_run"],
            "total_nights": self._total_nights,
            "rejected_nights": self._rejected_nights,
        }

    def _extract_chain_experiences(
        self,
        experiences: list[Experience],
    ) -> list[dict]:
        """
        从经验中提取因果链。

        Args:
            experiences: Buffer 中的经验

        Returns:
            因果链列表
        """
        chain = []
        for exp in experiences:
            chain.append({
                "embedding": exp.composite[:SSM_STATE_DIM] if exp.composite.shape[-1] >= SSM_STATE_DIM else exp.composite,
                "importance": getattr(exp, "importance", 0.0),
                "surprise": getattr(exp, "surprise", 0.0),
            })
        return chain

    def _post_process_buffer(
        self,
        experiences: list[Experience],
        result: dict[str, Any],
    ) -> None:
        """
        后处理 Buffer：更新经验元数据。

        Args:
            experiences: 已处理的经历
            result: 夜晚执行结果
        """
        # 更新经验的已整合标志
        for exp in experiences:
            if hasattr(exp, "consolidated"):
                exp.consolidated = True

    def get_statistics(self) -> dict[str, Any]:
        """
        获取夜晚流程统计信息。

        Returns:
            统计字典
        """
        acceptance_rate = 1.0
        if self._total_nights > 0:
            acceptance_rate = 1.0 - (self._rejected_nights / self._total_nights)

        return {
            "total_nights": self._total_nights,
            "rejected_nights": self._rejected_nights,
            "acceptance_rate": acceptance_rate,
            "current_night_count": self._night_count,
        }

    def reset(self) -> None:
        """重置夜晚流程状态"""
        self._night_count = 0
        self.controller.reset()
