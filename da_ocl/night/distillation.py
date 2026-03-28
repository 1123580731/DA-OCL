"""
DA-OCL 蒸馏模块
蒸馏损失路径：将快记忆的联想检索结果蒸馏到慢记忆
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da_ocl.core.constants import SSM_STATE_DIM


class DistillationPath(nn.Module):
    """
    蒸馏路径。

    将快记忆（Modern Hopfield Network）的联想检索结果蒸馏到慢记忆，
    使用加权 MSE 损失。
    """

    def __init__(self) -> None:
        super().__init__()
        # 轻量级适配层：将快记忆检索结果映射到慢记忆空间
        self.adapter = nn.Linear(SSM_STATE_DIM, SSM_STATE_DIM)
        nn.init.xavier_uniform_(self.adapter.weight, gain=0.5)
        nn.init.zeros_(self.adapter.bias)

    def forward(
        self,
        fast_retrieved: torch.Tensor,
        slow_state: torch.Tensor,
        importance: torch.Tensor,
        surprise: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算蒸馏损失。

        Args:
            fast_retrieved: (batch, ssm_state_dim) 快记忆检索结果
            slow_state: (batch, ssm_state_dim) 慢记忆当前状态
            importance: (batch,) 重要性分数
            surprise: (batch,) 惊喜度

        Returns:
            加权 MSE 蒸馏损失
        """
        # 适配快记忆输出到慢记忆空间
        adapted = self.adapter(fast_retrieved)

        # 加权 MSE
        mse = torch.mean((adapted - slow_state) ** 2, dim=-1)
        weight = importance * (1.0 + surprise)
        loss = (weight * mse).mean()

        return loss
