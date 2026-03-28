"""
DA-OCL 保护区
死亡经验 + 因果链追溯
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from da_ocl.buffer.experience import Experience
from da_ocl.core.constants import (
    DEATH_CAUSAL_CHAIN_T,
    DEATH_PRIORITY_GAMMA,
    DEATH_R_PENALTY_THRESHOLD,
)

if TYPE_CHECKING:
    from da_ocl.buffer.three_zone_buffer import ThreeZoneBuffer


class ProtectedZone:
    """
    保护区：不参与竞争替换的固定区域。

    替换条件（严格）：
    - 区域已满 AND 新死亡经验的 r_penalty 更高
    - 替换区内 priority_score 最低的经验

    行为：
    - 死亡经验直接进入保护区
    - 因果链追溯：死亡帧前 DEATH_CAUSAL_CHAIN_T 帧均进入保护区
    - 优先级随距离衰减：priority(frame_k) = r_penalty × gamma^(T_death - k)
    """

    def __init__(
        self,
        capacity: int,
        causal_chain_T: int = DEATH_CAUSAL_CHAIN_T,
        priority_gamma: float = DEATH_PRIORITY_GAMMA,
    ) -> None:
        self.capacity = capacity
        self.causal_chain_T = causal_chain_T
        self.priority_gamma = priority_gamma
        self._experiences: list[Experience] = []

    @property
    def size(self) -> int:
        return len(self._experiences)

    @property
    def is_full(self) -> bool:
        return len(self._experiences) >= self.capacity

    def add_death_experience(
        self,
        experience: Experience,
        causal_chain: list[Experience] | None = None,
    ) -> bool:
        """
        添加死亡经验及其因果链。

        Args:
            experience: 死亡帧经验（必须 is_death=True）
            causal_chain: 死亡前的经验列表（追溯 DEATH_CAUSAL_CHAIN_T 帧）

        Returns:
            True if successfully added
        """
        assert experience.is_death, "只能添加死亡经验到保护区"

        # 死亡帧本身
        death_exp = experience
        death_exp.priority_score = experience.r_penalty  # 最高优先级

        # 因果链追溯
        chain_experiences = []
        if causal_chain:
            # 优先级衰减：越接近死亡帧，优先级越高
            chain_len = min(len(causal_chain), self.causal_chain_T)
            for k, past_exp in enumerate(reversed(causal_chain[-chain_len:])):
                decay = self.priority_gamma ** (k + 1)
                past_exp.priority_score = experience.r_penalty * decay
                chain_experiences.append(past_exp)

        # 如果死亡发生在窗口内，整个窗口直接进保护区
        # （窗口本身已提供足够的因果上下文）

        # 尝试添加到缓冲区
        return self._try_add(death_exp) and all(
            self._try_add(exp) for exp in chain_experiences
        )

    def _try_add(self, experience: Experience) -> bool:
        """
        尝试添加经验到保护区。

        替换逻辑：
        1. 有空间 → 直接添加
        2. 满且新经验优先级更高 → 替换最低优先级
        3. 满且新经验优先级不高 → 拒绝
        """
        if len(self._experiences) < self.capacity:
            self._experiences.append(experience)
            return True

        # 满了：检查是否应该替换
        # min() 找 priority_score 最低的索引
        min_idx = min(
            range(len(self._experiences)),
            key=lambda i: self._experiences[i].priority_score,
        )

        # 严格小于才替换（相等时不替换）
        if experience.priority_score > self._experiences[min_idx].priority_score:
            self._experiences[min_idx] = experience
            return True

        return False

    def sample(self, num: int) -> list[Experience]:
        """
        采样经验用于夜晚蒸馏（按优先级降序）。

        Args:
            num: 采样数量

        Returns:
            经验列表
        """
        if not self._experiences:
            return []

        # 按优先级降序排列
        sorted_exps = sorted(self._experiences, key=lambda e: e.priority_score, reverse=True)
        return sorted_exps[:num]

    def get_state(self) -> dict:
        """返回保护区状态"""
        return {
            "experiences": [exp.to_dict() for exp in self._experiences],
            "size": self.size,
        }

    def load_state(self, state: dict) -> None:
        """从状态恢复"""
        self._experiences = [Experience.from_dict(d) for d in state["experiences"]]
