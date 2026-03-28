"""
DA-OCL 编码器层
将单帧原始观测 obs ∈ R^{obs_dim} 编码为实体特征向量 entity ∈ R^{entity_dim}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da_ocl.core.constants import ENTITY_DIM, GNN_HIDDEN_DIM, OBS_DIM


class Encoder(nn.Module):
    """
    观测编码器：将环境观测编码为实体特征向量。

    架构：obs_dim → 256 → 128（entity_dim），中间有 ReLU 激活和 Dropout。

    Args:
        obs_dim: 输入观测维度（默认 128，来自 constants.py）
        entity_dim: 输出实体嵌入维度（默认 128，来自 constants.py）

    Input shape:
        obs: (obs_dim,) 或 (batch, obs_dim)

    Output shape:
        entity: (entity_dim,) 或 (batch, entity_dim)

    Raises:
        AssertionError: 如果输入维度不匹配 obs_dim
    """

    obs_dim: int = OBS_DIM
    entity_dim: int = ENTITY_DIM

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        entity_dim: int = ENTITY_DIM,
        hidden_dim: int = GNN_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, entity_dim),
        )

        self._output_dim = entity_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (obs_dim,) 或 (batch, obs_dim)

        Returns:
            entity: (entity_dim,) 或 (batch, entity_dim)
        """
        if obs.dim() == 1:
            # 单帧输入：(obs_dim,) → (entity_dim,)
            assert obs.shape[-1] == self.obs_dim, (
                f"输入维度 {obs.shape[-1]} 与 obs_dim {self.obs_dim} 不匹配"
            )
            return self.net(obs)
        elif obs.dim() == 2:
            # 批次输入：(batch, obs_dim) → (batch, entity_dim)
            batch = obs.shape[0]
            assert obs.shape[1] == self.obs_dim, (
                f"输入维度 {obs.shape[1]} 与 obs_dim {self.obs_dim} 不匹配"
            )
            return self.net(obs)
        else:
            raise ValueError(f"obs 维度应为 1 或 2，实际为 {obs.dim()}")

    def get_output_dim(self) -> int:
        """返回 ENTITY_DIM（编译时验证）"""
        return self._output_dim
