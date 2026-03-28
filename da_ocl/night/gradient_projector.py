"""
DA-OCL 梯度投影模块
只更新掩码历史覆盖的参数，保护已学习的知识
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from da_ocl.core.constants import (
    GRADIENT_PROJECTION_EPS,
    SLOW_MEMORY_DIM,
)

if TYPE_CHECKING:
    from da_ocl.memory.mask_history import MaskHistory


class GradientProjector(nn.Module):
    """
    梯度投影器。

    作用：将梯度投影到掩码历史覆盖的参数子空间，
    只更新被日间高强度信号覆盖的参数。

    方法：
    1. 根据 mask_history 计算参数重要性
    2. 将梯度投影到重要性子空间
    3. 对不重要参数的梯度置零
    """

    def __init__(self, dim: int = SLOW_MEMORY_DIM) -> None:
        super().__init__()
        self.dim = dim

    def project(
        self,
        gradients: dict[str, torch.Tensor],
        mask_history: "MaskHistory",
    ) -> dict[str, torch.Tensor]:
        """
        将梯度投影到掩码覆盖的参数子空间。

        Args:
            gradients: {name: tensor} 当前梯度
            mask_history: 掩码历史

        Returns:
            投影后的梯度
        """
        projected = {}

        # 获取累积掩码强度
        accumulated = mask_history.get_accumulated()

        for name, grad in gradients.items():
            if grad is None:
                continue

            # 查找对应的掩码强度
            if name in accumulated:
                intensity = accumulated[name]
            else:
                # 未记录的参数，梯度清零
                intensity = torch.zeros_like(grad).mean()

            # 将强度映射为 [0, 1] 的掩码
            intensity_tensor = torch.tensor(intensity, device=grad.device, dtype=grad.dtype)
            mask = (intensity_tensor > GRADIENT_PROJECTION_EPS).float()

            # 投影：只保留掩码覆盖的梯度
            projected[name] = grad * mask

        return projected

    def compute_effective_lr(
        self,
        base_lr: float,
        mask_history: "MaskHistory",
        param_name: str,
    ) -> float:
        """
        根据掩码历史计算有效学习率。

        参数被覆盖强度越高 → 有效学习率越高（应该被更新）
        参数被覆盖强度越低 → 有效学习率越低（应该被保护）

        Args:
            base_lr: 基础学习率
            mask_history: 掩码历史
            param_name: 参数名称

        Returns:
            有效学习率
        """
        accumulated = mask_history.get_accumulated()
        if param_name in accumulated:
            intensity = accumulated[param_name].item()
            # 线性缩放
            return base_lr * intensity
        return 0.0  # 未被覆盖的参数不更新

    def apply_projected_gradients(
        self,
        model: nn.Module,
        projected_gradients: dict[str, torch.Tensor],
        lr: float,
    ) -> None:
        """
        将投影后的梯度应用到模型参数。

        Args:
            model: 慢记忆模型
            projected_gradients: 投影后的梯度
            lr: 学习率
        """
        for name, param in model.named_parameters():
            if name in projected_gradients:
                grad = projected_gradients[name]
                param.data.sub_(lr * grad)
