"""
DA-OCL 惊喜度检测器
计算窗口级别的惊喜度（慢记忆输出与检索结果的差异）
"""

from __future__ import annotations

import torch

from da_ocl.core.constants import COMPOSITE_DIM, SSM_STATE_DIM, WINDOW_SIZE_DEFAULT


class SurpriseDetector:
    """
    惊喜度检测器。

    计算当前窗口与慢记忆检索结果之间的惊喜度。
    Surp_batch = ||O_batch - O_hat_batch||_2

    惊喜度衡量「当前经验违反了已知规律的程度」。
    在 DA-OCL 中用于：联合掩码强度、情景边界检测、回放优先级、蒸馏权重。
    """

    def __init__(self) -> None:
        self._history: list[float] = []
        self._window_size: int = 20

    def update(self, surprise: float) -> None:
        """添加新的惊喜度值"""
        self._history.append(surprise)
        # 保持滑动窗口大小
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size:]

    def get_scalar(self) -> float:
        """获取当前平均惊喜度"""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def reset(self) -> None:
        """重置历史"""
        self._history = []

    @staticmethod
    def compute_window_surprise(
        o_batch: torch.Tensor,
        o_hat_batch: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """
        计算窗口级别的惊喜度。

        Args:
            o_batch: (SSM_STATE_DIM,) 当前窗口实际慢记忆表征
            o_hat_batch: (SSM_STATE_DIM,) 检索到的规律表征

        Returns:
            scalar: 窗口级别的惊喜度标量
            per_frame: (window_size,) 每帧的惊喜度（用于联合掩码的 max 计算）

        Raises:
            AssertionError: 如果 shape 不匹配
        """
        assert o_batch.shape[-1] == SSM_STATE_DIM, (
            f"o_batch 维度 {o_batch.shape[-1]} 不等于 SSM_STATE_DIM {SSM_STATE_DIM}"
        )
        assert o_hat_batch.shape[-1] == SSM_STATE_DIM, (
            f"o_hat_batch 维度 {o_hat_batch.shape[-1]} 不等于 SSM_STATE_DIM {SSM_STATE_DIM}"
        )

        diff = o_batch - o_hat_batch  # (SSM_STATE_DIM,)
        scalar = torch.norm(diff, p=2).item()

        # 简化实现：将等量惊喜度分配给每一帧
        # 完整实现可通过反向投影估算每帧贡献
        T = WINDOW_SIZE_DEFAULT
        per_frame = torch.full((T,), scalar / T, device=o_batch.device)

        return scalar, per_frame

    @staticmethod
    def compute_composite_surprise(
        composite_batch: torch.Tensor,
        composite_retrieved: torch.Tensor,
    ) -> float:
        """
        在复合特征空间计算惊喜度（作为补充指标）。

        Args:
            composite_batch: (COMPOSITE_DIM,) 当前窗口复合特征
            composite_retrieved: (COMPOSITE_DIM,) 快记忆检索到的复合特征

        Returns:
            float: 复合特征空间的惊喜度
        """
        assert composite_batch.shape[-1] == COMPOSITE_DIM
        assert composite_retrieved.shape[-1] == COMPOSITE_DIM
        diff = composite_batch - composite_retrieved
        return torch.norm(diff, p=2).item()
