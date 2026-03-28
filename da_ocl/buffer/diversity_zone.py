"""
DA-OCL 规律多样性区
模式分离保护：保证规律库稀疏区域始终有训练信号
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from da_ocl.buffer.experience import Experience
from da_ocl.core.constants import COMPOSITE_DIM

if TYPE_CHECKING:
    pass


class DiversityZone:
    """
    规律多样性区：模式分离保护。

    替换策略：
    - 新窗口进入时，计算其规律覆盖与区内已有经验的 overlap
    - overlap 高 → 丢弃（该规律域已有充分代表）
    - overlap 低 → 替换区内 overlap 最高的经验

    overlap 度量：复合特征的余弦相似度（代理变量）
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._experiences: list[Experience] = []

    @property
    def size(self) -> int:
        return len(self._experiences)

    @property
    def is_full(self) -> bool:
        return len(self._experiences) >= self.capacity

    def compute_overlap(
        self,
        new_composite: torch.Tensor,
        existing_composite: torch.Tensor,
    ) -> float:
        """
        计算两条经验的规律重叠度（余弦相似度）。

        使用 composite 特征的余弦相似度作为代理。

        Args:
            new_composite: (COMPOSITE_DIM,)
            existing_composite: (COMPOSITE_DIM,)

        Returns:
            float ∈ [-1, 1]，余弦相似度
        """
        n = F.normalize(new_composite.unsqueeze(0), dim=-1)  # (1, dim)
        e = F.normalize(existing_composite.unsqueeze(0), dim=-1)  # (1, dim)
        sim = F.cosine_similarity(n, e, dim=-1)
        return sim.item()

    def compute_max_overlap(self, new_exp: Experience) -> float:
        """
        计算新经验与区内所有经验的最大重叠度。

        Returns:
            float ∈ [0, 1]，最大重叠度（0 表示全新）
        """
        if not self._experiences:
            return 0.0

        max_sim = -1.0
        for exp in self._experiences:
            sim = self.compute_overlap(new_exp.composite, exp.composite)
            if sim > max_sim:
                max_sim = sim

        return max(max_sim, 0.0)

    def add(self, experience: Experience, overlap_threshold: float = 0.85) -> bool:
        """
        添加经验到多样性区。

        Args:
            experience: 待添加的经验
            overlap_threshold: 重叠度阈值，超过此值则丢弃

        Returns:
            True if added (或被拒绝/替换)
        """
        max_overlap = self.compute_max_overlap(experience)
        experience.pattern_overlap = max_overlap

        # 重叠度超过阈值 → 该规律域已有充分代表，丢弃
        if max_overlap >= overlap_threshold:
            return False

        # 重叠度低 → 可以添加
        if len(self._experiences) < self.capacity:
            self._experiences.append(experience)
            return True

        # 满了：替换区内重叠度最高的经验
        max_overlap_idx = max(
            range(len(self._experiences)),
            key=lambda i: self._experiences[i].pattern_overlap,
        )
        self._experiences[max_overlap_idx] = experience
        return True

    def sample(self, num: int) -> list[Experience]:
        """随机采样（保持多样性）"""
        if not self._experiences:
            return []
        import random
        return random.sample(self._experiences, min(num, len(self._experiences)))

    def resize(self, new_capacity: int) -> None:
        """
        调整容量。

        若缩小：从区内移除优先级最低的经验（最高 overlap）
        若扩大：直接接受，不影响现有经验

        Args:
            new_capacity: 新的容量上限
        """
        if new_capacity < self.capacity:
            # 需要缩小：移除重叠度最高（优先级最低）的经验
            while len(self._experiences) > new_capacity and self._experiences:
                max_idx = max(
                    range(len(self._experiences)),
                    key=lambda i: self._experiences[i].pattern_overlap,
                )
                self._experiences.pop(max_idx)
        self.capacity = new_capacity

    def get_state(self) -> dict:
        return {
            "experiences": [exp.to_dict() for exp in self._experiences],
            "size": self.size,
        }

    def load_state(self, state: dict) -> None:
        self._experiences = [Experience.from_dict(d) for d in state["experiences"]]
