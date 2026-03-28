"""
DA-OCL 三区 Buffer 容器
统一管理三个区域的动态容量分配和采样
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import torch

from da_ocl.buffer.diversity_zone import DiversityZone
from da_ocl.buffer.experience import Experience
from da_ocl.buffer.protected_zone import ProtectedZone
from da_ocl.buffer.task_zone import TaskOrientedZone
from da_ocl.core.constants import (
    BUFFER_ZONE_RATIOS_INITIAL,
    BUFFER_ZONE_RATIOS_MATURE,
)

if TYPE_CHECKING:
    pass


class ThreeZoneBuffer:
    """
    三区海马体启发 Buffer。

    各区独立管理，通过 capacity_allocator 动态调整容量分配。
    替换逻辑各区独立，不跨区竞争。

    容量分配策略：
    - 规律库越空（均值低）→ 多样性区占比越大
    - 规律库趋于饱和（均值高）→ 任务导向区占比越大
    """

    def __init__(self, total_capacity: int = 10000) -> None:
        self.total_capacity = total_capacity

        # 初始化各区容量（初期比例）
        self._init_ratios = BUFFER_ZONE_RATIOS_INITIAL
        self._mature_ratios = BUFFER_ZONE_RATIOS_MATURE

        self._protected = ProtectedZone(
            capacity=int(total_capacity * self._init_ratios["protected"]),
        )
        self._diversity = DiversityZone(
            capacity=int(total_capacity * self._init_ratios["diversity"]),
        )
        self._task = TaskOrientedZone(
            capacity=int(total_capacity * self._init_ratios["task"]),
        )

        self._is_mature: bool = False

    @property
    def protected(self) -> ProtectedZone:
        return self._protected

    @property
    def diversity(self) -> DiversityZone:
        return self._diversity

    @property
    def task(self) -> TaskOrientedZone:
        return self._task

    def total_window_count(self) -> int:
        """Buffer 中存储的窗口总数"""
        return self._protected.size + self._diversity.size + self._task.size

    def add(self, experience: Experience, zone_hint: Optional[str] = None) -> bool:
        """
        尝试添加一条经验到对应区域。

        Args:
            experience: 待添加的经验
            zone_hint: 可选的区域提示（"protected", "diversity", "task"）
                      如未指定，根据 experience.is_death 自动决定

        Returns:
            True if successfully added

        Raises:
            ValueError: zone_hint 不合法
        """
        # 死亡经验 → 保护区
        if experience.is_death:
            return self._protected.add_death_experience(experience, causal_chain=None)

        # 明确指定区域
        if zone_hint is not None:
            if zone_hint == "diversity":
                return self._diversity.add(experience)
            elif zone_hint == "task":
                return self._task.add(experience)
            else:
                raise ValueError(f"未知的 zone_hint: {zone_hint}")

        # 默认策略：根据规律重叠度选择区域
        max_overlap = self._diversity.compute_max_overlap(experience)
        if max_overlap < 0.85:
            # 多样性区有空间
            return self._diversity.add(experience)
        else:
            # 进入任务导向区
            return self._task.add(experience)

    def add_death(
        self,
        experience: Experience,
        causal_chain: list[Experience] | None = None,
    ) -> bool:
        """
        添加死亡经验到保护区（含因果链追溯）。

        Args:
            experience: 死亡帧经验（必须 is_death=True）
            causal_chain: 死亡前的经验列表

        Returns:
            True if successfully added
        """
        assert experience.is_death, "必须为死亡经验"
        return self._protected.add_death_experience(experience, causal_chain)

    def sample(
        self,
        num: int,
        priority: Literal["protected_first", "diversity", "task"] = "protected_first",
    ) -> list[Experience]:
        """
        按优先级采样经验用于夜晚蒸馏。

        Args:
            num: 采样数量
            priority: 采样优先级策略

        Returns:
            经验列表（可能少于 num，如果 Buffer 不足）
        """
        if priority == "protected_first":
            # 保护区优先 → 多样性区 → 任务导向区
            samples: list[Experience] = []
            remaining = num

            protected = self._protected.sample(remaining)
            samples.extend(protected)
            remaining -= len(protected)

            if remaining > 0:
                diversity = self._diversity.sample(remaining)
                samples.extend(diversity)
                remaining -= len(diversity)

            if remaining > 0:
                task = self._task.sample(remaining)
                samples.extend(task)

            return samples

        elif priority == "diversity":
            return self._diversity.sample(num)
        elif priority == "task":
            return self._task.sample(num)
        else:
            raise ValueError(f"未知的优先级策略: {priority}")

    def rebalance(self, coverage_global_mean: float) -> None:
        """
        根据规律库成熟度动态调整各区容量。

        规律库越空（均值低）→ 多样性区占比越大
        规律库趋于饱和（均值高）→ 任务导向区占比越大

        保护区容量固定为 10%，不参与动态调整。
        """
        self._is_mature = coverage_global_mean > 0.5

        # 计算目标容量
        if self._is_mature:
            target_diversity = int(self.total_capacity * self._mature_ratios["diversity"])
            target_task = int(self.total_capacity * self._mature_ratios["task"])
        else:
            target_diversity = int(self.total_capacity * self._init_ratios["diversity"])
            target_task = int(self.total_capacity * self._init_ratios["task"])

        target_protected = int(self.total_capacity * self._init_ratios["protected"])

        # 安全调整（只缩小，不因扩大而破坏已有经验）
        self._diversity.resize(min(target_diversity, self._diversity.capacity))
        self._task.resize(min(target_task, self._task.capacity))
        self._protected.capacity = target_protected

    @property
    def is_mature(self) -> bool:
        """规律库是否趋于成熟"""
        return self._is_mature

    def size(self) -> int:
        """返回 Buffer 中经验总数"""
        return (
            self._protected.size
            + self._diversity.size
            + self._task.size
        )

    def reset(self) -> None:
        """重置 Buffer（清空所有区域）"""
        self._protected = ProtectedZone(
            capacity=int(self.total_capacity * self._init_ratios["protected"]),
        )
        self._diversity = DiversityZone(
            capacity=int(self.total_capacity * self._init_ratios["diversity"]),
        )
        self._task = TaskOrientedZone(
            capacity=int(self.total_capacity * self._init_ratios["task"]),
        )

    def get_all_experiences(self) -> list[Experience]:
        """返回所有区域的 Experience 列表（供夜晚蒸馏使用）"""
        return (
            list(self._protected._experiences)
            + list(self._diversity._experiences)
            + list(self._task._experiences)
        )

    def load_state(self, state: dict) -> None:
        """从 checkpoint 恢复"""
        self._protected.load_state(state["protected"])
        self._diversity.load_state(state["diversity"])
        self._task.load_state(state["task"])
        self._is_mature = state.get("is_mature", False)

    def get_checkpoint_state(self) -> dict:
        """返回完整 Buffer 状态（用于 checkpoint 保存）"""
        return {
            "protected": self._protected.get_state(),
            "diversity": self._diversity.get_state(),
            "task": self._task.get_state(),
            "total_capacity": self.total_capacity,
            "is_mature": self._is_mature,
        }
