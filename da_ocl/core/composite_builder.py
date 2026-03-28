"""
DA-OCL 复合特征构建器
B = concat(O_batch, O_hat_batch, O_batch - O_hat_batch)
dim = 3 × ssm_state_dim = 1536
"""

from __future__ import annotations

import torch

from da_ocl.core.constants import COMPOSITE_DIM, SSM_STATE_DIM


class CompositeFeatureBuilder:
    """
    构建因果窗口级别的复合特征。

    公式：B = concat(O_batch, O_hat_batch, O_batch - O_hat_batch)
    维度：(3 × SSM_STATE_DIM) = 1536

    语义：
    - O_batch：当前窗口的实际慢记忆输出
    - O_hat_batch：检索到的规律激活向量
    - O - O_hat：残差，定位「哪些时刻、哪些实体关系违反了已知规律」

    认知冲突的定位精度从帧级别提升到事件级别。
    """

    @staticmethod
    def build(
        o_batch: torch.Tensor,
        o_hat_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        构建复合特征。

        Args:
            o_batch: (SSM_STATE_DIM,) 当前窗口实际慢记忆表征
            o_hat_batch: (SSM_STATE_DIM,) 检索到的规律表征

        Returns:
            composite: (COMPOSITE_DIM,) 其中 COMPOSITE_DIM = 1536

        Raises:
            AssertionError: shape 不符合契约
        """
        assert o_batch.shape[-1] == SSM_STATE_DIM, (
            f"o_batch 维度 {o_batch.shape[-1]} 不等于 SSM_STATE_DIM {SSM_STATE_DIM}"
        )
        assert o_hat_batch.shape[-1] == SSM_STATE_DIM, (
            f"o_hat_batch 维度 {o_hat_batch.shape[-1]} 不等于 SSM_STATE_DIM {SSM_STATE_DIM}"
        )

        residual = o_batch - o_hat_batch
        composite = torch.cat([o_batch, o_hat_batch, residual], dim=-1)

        assert composite.shape[-1] == COMPOSITE_DIM, (
            f"复合特征维度 {composite.shape[-1]} 不等于 COMPOSITE_DIM {COMPOSITE_DIM}"
        )

        return composite

    @staticmethod
    def decompose(
        composite: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从复合特征反向分解。

        Args:
            composite: (COMPOSITE_DIM,) 复合特征

        Returns:
            (o_batch, o_hat_batch, residual)
        """
        assert composite.shape[-1] == COMPOSITE_DIM
        o_batch = composite[:SSM_STATE_DIM]
        o_hat_batch = composite[SSM_STATE_DIM : 2 * SSM_STATE_DIM]
        residual = composite[2 * SSM_STATE_DIM :]
        return o_batch, o_hat_batch, residual
