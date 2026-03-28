"""
DA-OCL 慢记忆系统
三层因果结构编码器（GN + SSM + Rules）
支持沉默写入（白天）和检索路径
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from da_ocl.core.constants import (
    ENTITY_DIM,
    GNN_HIDDEN_DIM,
    RULE_DIM,
    SSM_STATE_DIM,
)
from da_ocl.core.encoder import Encoder
from da_ocl.core.gnn_layer import GNNRelationEncoder
from da_ocl.core.ssm_layer import SSMCompressor
from da_ocl.core.rule_constraints import RuleConstraintLayer


class SlowMemory(nn.Module):
    """
    慢记忆三层结构容器。

    白天：因果窗口 → GNN → SSM → Rules → 表征输出
    白天同步：沉默写入（受联合掩码控制）
    白天检索：返回规律激活向量 O_hat

    三层串联：
    1. GNN 实体关系层：实体关系动态变化
    2. SSM 规律压缩层：跨帧时序不变规律
    3. 逻辑约束层：元规则软约束

    Args:
        entity_dim: 实体嵌入维度（默认 128）
        gnn_hidden_dim: GNN 中间层维度（默认 256）
        ssm_state_dim: SSM 隐状态维度（默认 512）
        rule_dim: 元规则维度（默认 256）
        use_mamba: 是否使用 mamba-ssm（默认 True，失败自动降级）
        num_entities: 实体数量（默认 4）
    """

    def __init__(
        self,
        entity_dim: int = ENTITY_DIM,
        gnn_hidden_dim: int = GNN_HIDDEN_DIM,
        ssm_state_dim: int = SSM_STATE_DIM,
        rule_dim: int = RULE_DIM,
        use_mamba: bool = True,
        num_entities: int = 4,
    ) -> None:
        super().__init__()

        self.gnn = GNNRelationEncoder(
            entity_dim=entity_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_entities=num_entities,
        )
        self.ssm = SSMCompressor(
            gnn_hidden_dim=gnn_hidden_dim,
            ssm_state_dim=ssm_state_dim,
            use_mamba=use_mamba,
        )
        self.rules = RuleConstraintLayer(
            ssm_state_dim=ssm_state_dim,
            rule_dim=rule_dim,
        )

    def forward(
        self,
        entity_window: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        三层级联前向（白天感知路径）。

        Args:
            entity_window: (T, entity_dim) 因果窗口的实体嵌入序列

        Returns:
            gnn_out: (T, GNN_HIDDEN_DIM) 跨帧实体关系表征
            ssm_out: (SSM_STATE_DIM,) 规律压缩表征（O_batch）
            rule_out: (RULE_DIM,) 元规则约束后的慢记忆表征
        """
        gnn_out = self.gnn(entity_window)         # (T, GNN_HIDDEN_DIM)
        ssm_out = self.ssm(gnn_out)               # (SSM_STATE_DIM,)
        rule_out = self.rules(ssm_out)             # (RULE_DIM,)
        return gnn_out, ssm_out, rule_out

    def retrieve(self, entity_window: torch.Tensor) -> torch.Tensor:
        """
        慢记忆检索路径（白天白天白天白天白天白天白天）。

        直接返回 SSM 压缩表征作为规律检索结果。
        简化实现：完整实现应包含规律库索引和相似度检索。

        Args:
            entity_window: (T, entity_dim) 因果窗口

        Returns:
            o_hat: (SSM_STATE_DIM,) 规律激活向量（检索到的规律表征）
        """
        _, ssm_out, _ = self.forward(entity_window)
        return ssm_out

    def silent_write(
        self,
        entity_window: torch.Tensor,
        mask_value: float,
    ) -> None:
        """
        沉默写入（白天联合掩码控制）。

        根据联合掩码值，对慢记忆参数进行受控的弱更新。
        掩码值越大 → 写入强度越大。
        掩码值接近 0 时，参数几乎不变。

        Args:
            entity_window: (T, entity_dim) 当前因果窗口
            mask_value: 联合掩码值 ∈ [0, 1]
        """
        if mask_value <= 0.0:
            return  # 掩码为 0，不写入

        # 获取当前输出作为"理想目标"
        _, ssm_out, _ = self.forward(entity_window)

        # 沉默写入策略：
        # 不执行实际的梯度更新（这是白天的行为），
        # 仅记录需要更新的参数位置和强度（供夜晚激活使用）
        # 实际更新在夜晚阶段执行。
        # 这里将 mask_value 附加到参数的 grad_info 中。
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # 创建一个影子属性存储掩码强度
                    if not hasattr(self, "_mask_intensity"):
                        self._mask_intensity: dict[str, float] = {}
                    # 取 max（与联合掩码的 max_t 一致）
                    current = self._mask_intensity.get(name, 0.0)
                    self._mask_intensity[name] = max(current, mask_value)

    def get_mask_intensity(self) -> dict[str, float]:
        """获取各参数的掩码强度（供夜晚激活索引使用）"""
        if hasattr(self, "_mask_intensity"):
            return self._mask_intensity.copy()
        return {}

    def clear_mask_intensity(self) -> None:
        """清空掩码强度记录（夜晚整合完成后调用）"""
        if hasattr(self, "_mask_intensity"):
            self._mask_intensity.clear()

    def get_state_dict(self) -> dict:
        """获取慢记忆完整状态（用于双缓冲）"""
        state = super().state_dict()
        if hasattr(self, "_mask_intensity"):
            state["_mask_intensity"] = self._mask_intensity.copy()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """从状态字典恢复慢记忆"""
        mask_intensity = state_dict.pop("_mask_intensity", None)
        super().load_state_dict(state_dict)
        if mask_intensity is not None:
            self._mask_intensity = mask_intensity
