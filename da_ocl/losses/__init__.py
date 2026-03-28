"""
DA-OCL 损失函数集合
"""

from __future__ import annotations

from da_ocl.losses.night_losses import (
    DistillationLoss,
    KLLoss,
    EWCLoss,
    DreamLoss,
    compute_total_night_loss,
)

__all__ = [
    "DistillationLoss",
    "KLLoss",
    "EWCLoss",
    "DreamLoss",
    "compute_total_night_loss",
]
