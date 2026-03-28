"""
DA-OCL 情景边界检测器
batch惊喜度突变检测，触发新情景开始
边界触发: mean(Surp_batch) > mu_history + N * sigma_history
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from da_ocl.core.constants import BOUNDARY_HISTORY_SIZE, BOUNDARY_N_SIGMA


class BoundaryDetector:
    """
    情景边界检测器。

    在 batch 粒度上检测惊喜度突变。
    比单帧粒度更稳定，减少噪声引起的假阳性。

    触发条件：
        mean(Surp_batch) > mu_history + N * sigma_history

    情景块是 Buffer 存储和替换的基本单位，不拆散单条经验。
    """

    def __init__(
        self,
        n_sigma: float = BOUNDARY_N_SIGMA,
        history_size: int = BOUNDARY_HISTORY_SIZE,
    ) -> None:
        self.n_sigma = n_sigma
        self.history_size = history_size
        self._history: deque[float] = deque(maxlen=history_size)
        self._is_new_boundary: bool = False
        self._boundary_count: int = 0

    def detect(self, batch_surprise: float) -> bool:
        """
        检测是否触发情景边界。

        Args:
            batch_surprise: float 窗口级别的惊喜度

        Returns:
            True: 触发新情景边界
            False: 同一情景继续

        Note:
            预热期（前 10 个 batch）不触发边界（历史数据不足）
        """
        self._history.append(batch_surprise)

        # 预热期不触发
        if len(self._history) < 10:
            self._is_new_boundary = False
            return False

        mu = np.mean(self._history)
        sigma = np.std(self._history) if len(self._history) > 1 else 1.0

        threshold = mu + self.n_sigma * sigma
        is_boundary = batch_surprise > threshold

        if is_boundary:
            self._boundary_count += 1
            self._is_new_boundary = True
        else:
            self._is_new_boundary = False

        return is_boundary

    def is_new_boundary(self) -> bool:
        """上一次检测是否为新边界（供后续流程查询）"""
        return self._is_new_boundary

    @property
    def boundary_count(self) -> int:
        """历史中触发的边界总数"""
        return self._boundary_count

    @property
    def history_mean(self) -> float:
        return np.mean(self._history) if self._history else 0.0

    @property
    def history_std(self) -> float:
        return np.std(self._history) if len(self._history) > 1 else 0.0

    def reset(self) -> None:
        """重置边界检测状态"""
        self._history.clear()
        self._is_new_boundary = False
        self._boundary_count = 0
