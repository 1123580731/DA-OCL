"""
DA-OCL 夜晚触发条件检测器
"""

from __future__ import annotations

from da_ocl.core.constants import (
    NIGHT_MASK_DENSITY_THRESHOLD,
    NIGHT_SURP_THRESHOLD,
    NIGHT_TIMESTEP_FALLBACK,
    NIGHT_WINDOW_THRESHOLD,
)


class NightTrigger:
    """
    夜晚触发条件检测器。

    四条触发条件（任一满足）：
    1. Buffer 窗口数量达到阈值
    2. 掩码激活密度超过阈值
    3. 滑动窗口平均 Surp 超过阈值
    4. 时间步数达到兜底阈值
    """

    def __init__(
        self,
        window_threshold: int = NIGHT_WINDOW_THRESHOLD,
        mask_density_threshold: float = NIGHT_MASK_DENSITY_THRESHOLD,
        surp_threshold: float = NIGHT_SURP_THRESHOLD,
        timestep_fallback: int = NIGHT_TIMESTEP_FALLBACK,
    ) -> None:
        self.window_threshold = window_threshold
        self.mask_density_threshold = mask_density_threshold
        self.surp_threshold = surp_threshold
        self.timestep_fallback = timestep_fallback

    def should_trigger(
        self,
        buffer_window_count: int,
        mask_density: float,
        avg_surprise: float,
        timestep: int,
    ) -> tuple[bool, str]:
        """
        检测是否应触发夜晚。

        Args:
            buffer_window_count: Buffer 中存储的窗口数量
            mask_density: 掩码激活密度 ∈ [0, 1]
            avg_surprise: 滑动窗口平均惊喜度
            timestep: 当前时间步数

        Returns:
            (should_trigger, reason) — 是否触发 + 触发原因
        """
        # 条件 1: Buffer 窗口数量达到阈值
        if buffer_window_count >= self.window_threshold:
            return True, f"Buffer 窗口数量达到阈值 ({buffer_window_count} >= {self.window_threshold})"

        # 条件 2: 掩码激活密度超过阈值
        if mask_density >= self.mask_density_threshold:
            return True, f"掩码激活密度超过阈值 ({mask_density:.3f} >= {self.mask_density_threshold})"

        # 条件 3: 平均惊喜度超过阈值
        if avg_surprise >= self.surp_threshold:
            return True, f"平均惊喜度超过阈值 ({avg_surprise:.3f} >= {self.surp_threshold})"

        # 条件 4: 时间步数达到兜底阈值
        if timestep >= self.timestep_fallback:
            return True, f"时间步数达到兜底阈值 ({timestep} >= {self.timestep_fallback})"

        return False, ""

    def reset_counters(self) -> None:
        """重置触发计数器（夜晚整合完成后调用）"""
        pass
