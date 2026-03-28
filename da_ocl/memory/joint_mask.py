"""
DA-OCL 联合掩码计算
mask_batch = max_t(sigmoid(Surp_t)) × (1 - CoverageRate_batch)
"""

from __future__ import annotations

import torch


def compute_joint_mask(
    surprise_per_frame: torch.Tensor,
    coverage_rate: float,
) -> float:
    """
    计算联合掩码。

    mask_batch = max_t(sigmoid(Surp_t)) × (1 - CoverageRate_batch)

    语义：
    - max_t(sigmoid(Surp_t))：窗口内峰值认知冲突强度。
      窗口内只要有一帧足够显著，整个窗口的写入强度都提升。
      这对应杏仁核对情景块的整体标记机制。
    - (1 - CoverageRate_batch)：当前窗口的规律覆盖缺口。
      规律库越稀疏的区域越开放写入，防止已知规律被过度强化。

    Args:
        surprise_per_frame: (T,) 每帧惊喜度（应在 [0, +∞) 范围）
        coverage_rate: float ∈ [0, 1]，规律库的平均覆盖率

    Returns:
        mask_scalar: float ∈ [0, 1]，联合掩码标量值

    Raises:
        ValueError: coverage_rate 不在 [0,1] 范围内
    """
    if not (0.0 <= coverage_rate <= 1.0):
        raise ValueError(
            f"coverage_rate {coverage_rate} 不在 [0, 1] 范围内"
        )

    # max_t(sigmoid(Surp_t))
    # 使用 ReLU-style：sigmoid(Surp) - 0.5，确保 Surp=0 时输出为 0
    # 这对应语义：零惊喜度 → 无认知冲突 → 不写入
    # 公式：max(sigmoid(Surp) - 0.5, 0) 使 Surp=0 时贡献为 0
    raw_conflict = torch.sigmoid(surprise_per_frame) - 0.5
    # clamp 确保非负，然后取 max
    peak_conflict = torch.clamp_min(raw_conflict, 0.0).max().item()

    # (1 - CoverageRate_batch)
    coverage_gap = 1.0 - coverage_rate

    # 联合掩码
    mask = peak_conflict * coverage_gap

    return float(mask)


def compute_joint_mask_batch(
    surprise_per_frame: torch.Tensor,
    coverage_rate: float,
) -> torch.Tensor:
    """
    批量计算联合掩码（返回张量版本，供梯度计算使用）。

    Args:
        surprise_per_frame: (T,) 或 (batch, T)
        coverage_rate: float 或 (batch,)

    Returns:
        mask: (batch,) 或 scalar，与参数张量同 shape 的掩码（标量广播）
    """
    raw_conflict = torch.sigmoid(surprise_per_frame) - 0.5
    peak_conflict = torch.clamp_min(raw_conflict, 0.0).max(dim=-1).values  # (batch,)
    coverage_gap = 1.0 - coverage_rate
    return peak_conflict * coverage_gap


class JointMaskComputer:
    """
    联合掩码计算器（有状态版本）。

    维护内部状态，支持滑动窗口内的联合掩码跟踪。
    """

    def __init__(self) -> None:
        self._coverage_rate: float = 0.5  # 默认 50% 覆盖率

    def compute_joint_mask(
        self,
        surprise: float,
        coverage_rate: float | None = None,
    ) -> float:
        """
        计算联合掩码（标量接口）。

        Args:
            surprise: 惊喜度标量
            coverage_rate: 可选，覆盖率（默认使用内部状态）

        Returns:
            联合掩码标量
        """
        if coverage_rate is not None:
            self._coverage_rate = coverage_rate

        # 将标量 surprise 包装为张量
        surprise_tensor = torch.tensor([surprise])
        return compute_joint_mask(surprise_tensor, self._coverage_rate)

    def set_coverage_rate(self, coverage_rate: float) -> None:
        """设置覆盖率"""
        self._coverage_rate = coverage_rate

    def get_coverage_rate(self) -> float:
        """获取当前覆盖率"""
        return self._coverage_rate
