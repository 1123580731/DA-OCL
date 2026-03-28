"""
DA-OCL 对抗滤波模块
两级滤波：batch 整体粗筛 + 逐帧细筛
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from da_ocl.core.constants import FILTER_COARSE_THRESHOLD, FILTER_FINE_THRESHOLD, FILTER_SKIP_FINE_IF_COARSE_ABOVE


@dataclass
class FilterResult:
    """滤波结果"""
    passed: bool
    coarse_passed: bool
    fine_passed: bool
    coarse_filter_ratio: float
    fine_filter_ratio: float | None  # None if skipped
    valid_frames_mask: torch.Tensor | None  # None if batch rejected


class AdversarialFilter:
    """
    两级对抗滤波。

    第一级（粗筛）：基于 batch 整体统计量，高冲突 batch 直接放弃。
    第二级（细筛）：仅在粗筛通过且过滤比 ≤ 50% 时执行，逐帧分析。

    决策逻辑：
    - 粗筛过滤比 > 50% → 跳过细筛，保留整个 batch（防止过度过滤）
    - 粗筛过滤比 ≤ 50% → 进入细筛
    - 细筛后有效帧数 < 2 → 放弃整个 batch
    """

    def __init__(
        self,
        coarse_threshold: float = FILTER_COARSE_THRESHOLD,
        fine_threshold: float = FILTER_FINE_THRESHOLD,
        skip_fine_if_coarse_above: float = FILTER_SKIP_FINE_IF_COARSE_ABOVE,
    ) -> None:
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.skip_fine_if_coarse_above = skip_fine_if_coarse_above

    def filter(
        self,
        composite: torch.Tensor,
        surprise_per_frame: torch.Tensor,
    ) -> FilterResult:
        """
        执行两级滤波。

        Args:
            composite: (3*SSM_STATE_DIM,) 复合特征
            surprise_per_frame: (T,) 每帧的惊喜度

        Returns:
            FilterResult
        """
        T = surprise_per_frame.shape[0]

        # === 第一级：粗筛 ===
        # 使用惊喜度的均值作为 batch 整体冲突指标
        mean_surprise = surprise_per_frame.mean().item()
        coarse_passed = mean_surprise <= self.coarse_threshold

        # 粗筛过滤比：超过阈值的程度
        coarse_filter_ratio = max(0.0, mean_surprise - self.coarse_threshold) / max(1e-6, mean_surprise)

        fine_passed = True
        fine_filter_ratio = None
        valid_frames_mask = None

        if coarse_passed:
            # 粗筛通过：进入第二级细筛
            if coarse_filter_ratio <= self.skip_fine_if_coarse_above:
                # 过滤比不高：进入细筛
                fine_passed, valid_frames_mask, fine_filter_ratio = self._fine_filter(
                    composite, surprise_per_frame
                )
        else:
            # 粗筛未通过
            if coarse_filter_ratio <= self.skip_fine_if_coarse_above:
                # 但过滤比不高：给细筛机会
                fine_passed, valid_frames_mask, fine_filter_ratio = self._fine_filter(
                    composite, surprise_per_frame
                )
                # 如果细筛找到足够多有效帧，可以挽救
                if valid_frames_mask is not None and valid_frames_mask.sum().item() < 2:
                    fine_passed = False
            else:
                # 粗筛过滤比太高：直接放弃
                fine_passed = False

        # 最终决策
        passed = coarse_passed or fine_passed

        return FilterResult(
            passed=passed,
            coarse_passed=coarse_passed,
            fine_passed=fine_passed,
            coarse_filter_ratio=coarse_filter_ratio,
            fine_filter_ratio=fine_filter_ratio,
            valid_frames_mask=valid_frames_mask,
        )

    def _fine_filter(
        self,
        composite: torch.Tensor,
        surprise_per_frame: torch.Tensor,
    ) -> tuple[bool, torch.Tensor | None, float | None]:
        """
        第二级细筛：逐帧分析。

        放弃高冲突帧，保留其余。

        Returns:
            (passed, valid_frames_mask, filter_ratio)
        """
        T = surprise_per_frame.shape[0]

        # 每帧冲突度（简化：均匀分配复合特征的 L2 范数）
        composite_norm = torch.norm(composite).item()
        per_frame_conflict = composite_norm * torch.ones(T, device=surprise_per_frame.device) / T

        # 高于细筛阈值的帧被标记为无效
        high_conflict_mask = per_frame_conflict > self.fine_threshold * surprise_per_frame.mean()
        valid_frames_mask = ~high_conflict_mask

        num_valid = valid_frames_mask.sum().item()
        fine_filter_ratio = (T - num_valid) / max(1, T)

        # 有效帧数 < 2 → 放弃
        passed = num_valid >= 2

        return passed, valid_frames_mask, fine_filter_ratio

    def filter_batch(
        self,
        batch_composites: torch.Tensor,
        batch_surprises: torch.Tensor,
    ) -> list[FilterResult]:
        """
        批量滤波（一个 episode 的多个 batch）。

        Args:
            batch_composites: (num_batches, COMPOSITE_DIM)
            batch_surprises: (num_batches, T)

        Returns:
            list of FilterResult
        """
        results = []
        for i in range(batch_composites.shape[0]):
            result = self.filter(batch_composites[i], batch_surprises[i])
            results.append(result)
        return results
