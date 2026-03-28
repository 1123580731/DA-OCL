"""
DA-OCL EWC 更新器
维护 Fisher 信息矩阵，实现弹性权重巩固
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from da_ocl.core.constants import (
    EWC_FISHER_SAMPLES,
    EWC_LAMBDA,
    SLOW_MEMORY_DIM,
)


class EWCUpdater(nn.Module):
    """
    EWC 更新器。

    维护 Fisher 信息矩阵，记录每个任务学习后的参数位置，
    在新任务学习时添加正则化项，保护已学习的知识。
    """

    def __init__(
        self,
        dim: int = SLOW_MEMORY_DIM,
        ewc_lambda: float = EWC_LAMBDA,
        fisher_samples: int = EWC_FISHER_SAMPLES,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples

        # 存储旧参数和 Fisher 信息
        self._old_params: dict[str, torch.Tensor] = {}
        self._fisher_diagonal: dict[str, torch.Tensor] = {}

    def snapshot(self, model: nn.Module) -> None:
        """
        拍摄参数快照（在任务完成后调用）。

        记录当前参数位置和 Fisher 信息。

        Args:
            model: 慢记忆模型
        """
        # 记录旧参数
        self._old_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    def compute_fisher(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> None:
        """
        计算 Fisher 信息矩阵对角线。

        Fisher 信息 ≈ 参数梯度平方的期望
        表示参数在似然函数中的敏感性。

        Args:
            model: 慢记忆模型
            inputs: 用于 Fisher 估计的输入数据
        """
        model.eval()
        self._fisher_diagonal = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
        }

        # 多次采样估计 Fisher 信息
        for _ in range(self.fisher_samples):
            model.zero_grad()
            output = model(inputs)
            # 使用输出负熵作为代理（简化实现）
            loss = -output.mean()
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    self._fisher_diagonal[n] += p.grad.detach() ** 2

        # 平均
        for n in self._fisher_diagonal:
            self._fisher_diagonal[n] /= float(self.fisher_samples)

        model.train()

    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        计算 EWC 正则化损失。

        L_EWC = lambda * Σ_i F_i * (theta_i - theta_i_star)^2

        Args:
            model: 慢记忆模型（当前参数）

        Returns:
            EWC 损失标量
        """
        if not self._old_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = 0.0
        for n, param in model.named_parameters():
            if n in self._old_params:
                fisher = self._fisher_diagonal.get(n)
                if fisher is not None:
                    diff = (param - self._old_params[n]) ** 2
                    loss += (fisher * diff).sum()

        return self.ewc_lambda * loss

    def merge_fisher(self, other: "EWCUpdater") -> None:
        """
        合并另一个 EWCUpdater 的 Fisher 信息。

        用于累积多个任务的保护信号。

        Args:
            other: 另一个 EWCUpdater
        """
        for n in self._fisher_diagonal:
            if n in other._fisher_diagonal:
                # 叠加 Fisher 信息
                self._fisher_diagonal[n] = (
                    self._fisher_diagonal[n] + other._fisher_diagonal[n]
                ) / 2.0

    def get_consolidated_params(self) -> dict[str, torch.Tensor]:
        """
        获取已巩固的参数快照。

        Returns:
            {name: param} 参数字典
        """
        return self._old_params.copy()
