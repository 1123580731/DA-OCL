"""
DA-OCL 任务导向区
目标导向显著性：支持经验的任务价值分数重新评估
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

from da_ocl.buffer.experience import Experience

if TYPE_CHECKING:
    pass


class TaskOrientedZone:
    """
    任务导向区：目标导向显著性。

    - 按 task_value 分数排序
    - 新经验进入时替换区内 task_value 最低的经验
    - 区内经验支持重新评分（同一个经验在不同任务目标下分数可以差很多）
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._experiences: list[Experience] = []
        self._sorted: bool = False

    @property
    def size(self) -> int:
        return len(self._experiences)

    @property
    def is_full(self) -> bool:
        return len(self._experiences) >= self.capacity

    def _ensure_sorted(self) -> None:
        """确保经验按 task_value 降序排列"""
        if not self._sorted:
            self._experiences.sort(key=lambda e: e.task_value, reverse=True)
            self._sorted = True

    def add(self, experience: Experience) -> bool:
        """
        添加经验到任务导向区。

        Args:
            experience: 待添加的经验

        Returns:
            True if successfully added
        """
        self._ensure_sorted()

        if len(self._experiences) < self.capacity:
            self._experiences.append(experience)
            self._sorted = False  # 需要重新排序
            return True

        # 满了：替换最低分的
        min_idx = min(range(len(self._experiences)), key=lambda i: self._experiences[i].task_value)
        if experience.task_value > self._experiences[min_idx].task_value:
            self._experiences[min_idx] = experience
            self._sorted = False
            return True

        return False

    def re_evaluate_all(self, new_task_weights: Dict[str, float]) -> None:
        """
        根据新的任务目标权重重新评估所有经验的任务价值。

        重新排序后，被低分经验占据的容量可以被新经验替换。

        Note:
            当前简化实现：所有经验的任务价值乘以权重因子
            完整实现应根据具体任务重新计算每个经验的 task_value
        """
        weight_factor = new_task_weights.get("global", 1.0)
        for exp in self._experiences:
            exp.task_value *= weight_factor
        self._sorted = False
        self._ensure_sorted()

    def sample(self, num: int) -> list[Experience]:
        """
        采样经验用于夜晚蒸馏（按 task_value 降序）。

        Args:
            num: 采样数量

        Returns:
            经验列表（按优先级降序）
        """
        self._ensure_sorted()
        return self._experiences[:num]

    def get_lowest_task_value(self) -> float:
        """获取当前最低任务价值（用于判断新经验是否值得替换）"""
        self._ensure_sorted()
        if not self._experiences:
            return 0.0
        return self._experiences[-1].task_value

    def resize(self, new_capacity: int) -> None:
        """
        调整容量。

        若缩小：从区内移除优先级最低的经验（最低 task_value）
        若扩大：直接接受，不影响现有经验

        Args:
            new_capacity: 新的容量上限
        """
        if new_capacity < self.capacity:
            # 需要缩小：移除任务价值最低（优先级最低）的经验
            self._ensure_sorted()
            while len(self._experiences) > new_capacity and self._experiences:
                self._experiences.pop()  # 已按 task_value 降序排列，最后一个最低
                self._sorted = True  # 保持排序状态
        self.capacity = new_capacity

    def get_state(self) -> dict:
        return {
            "experiences": [exp.to_dict() for exp in self._experiences],
            "size": self.size,
        }

    def load_state(self, state: dict) -> None:
        self._experiences = [Experience.from_dict(d) for d in state["experiences"]]
        self._sorted = False
