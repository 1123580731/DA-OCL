"""
DA-OCL GNN 实体关系编码层
处理实体节点之间的动态关系，捕捉动作驱动的实体关系演化
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from da_ocl.core.constants import ENTITY_DIM, GNN_HIDDEN_DIM


class GNNRelationEncoder(nn.Module):
    """
    GNN 实体关系编码层。

    接收因果窗口内每帧的实体特征，输出跨帧的实体关系表征序列。
    以可微连续表征存储符号关系，使规律间可进行向量运算。

    输入: entity_window ∈ R^{T x entity_dim}
    输出: relational_features ∈ R^{T x GNN_HIDDEN_DIM}

    跨帧建模策略：将 T 维度作为 batch 维度处理，建模跨帧关系变化。

    Args:
        entity_dim: 实体嵌入维度（默认 128）
        gnn_hidden_dim: GNN 中间层维度（默认 256）
        num_entities: 窗口内实体数量（默认 4）
        num_heads: 注意力头数（默认 4）
    """

    def __init__(
        self,
        entity_dim: int = ENTITY_DIM,
        gnn_hidden_dim: int = GNN_HIDDEN_DIM,
        num_entities: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.entity_dim = entity_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_entities = num_entities
        self.num_heads = num_heads

        # 每帧内的实体关系编码
        self.entity_proj = nn.Linear(entity_dim, gnn_hidden_dim)

        # 跨帧关系建模：GAT-lite（单层自注意力）
        assert gnn_hidden_dim % num_heads == 0, (
            f"gnn_hidden_dim {gnn_hidden_dim} 必须能被 num_heads {num_heads} 整除"
        )
        self.head_dim = gnn_hidden_dim // num_heads
        self.q_proj = nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        self.k_proj = nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        self.v_proj = nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        self.out_proj = nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        self.norm = nn.LayerNorm(gnn_hidden_dim)

        # 跨帧时序整合：将所有帧的 GNN 输出汇总
        self.temporal_encoder = nn.GRU(
            gnn_hidden_dim,
            gnn_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_proj = nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim)

    def forward(
        self,
        entity_window: torch.Tensor,
        entity_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            entity_window: (T, entity_dim) 窗口内实体的嵌入序列
                          或者 (T, num_entities, entity_dim) 如果有多实体
            entity_positions: (T, num_entities, 2) 每帧的实体位置（可选，暂未使用）

        Returns:
            relational_features: (T, GNN_HIDDEN_DIM) 跨帧实体关系表征
        """
        T = entity_window.shape[0]

        if entity_window.dim() == 2:
            # (T, entity_dim) → 扩展为 (T, num_entities, entity_dim)
            # 假设 num_entities 个实体共享同一嵌入（简化假设）
            entity_window = entity_window.unsqueeze(1)  # (T, 1, entity_dim)
            entity_window = entity_window.expand(-1, self.num_entities, -1)  # (T, num_entities, entity_dim)

        assert entity_window.dim() == 3, (
            f"entity_window 应为 (T, entity_dim) 或 (T, num_entities, entity_dim)，"
            f"实际为 {entity_window.shape}"
        )
        T, N, E = entity_window.shape  # N = num_entities, E = entity_dim
        assert E == self.entity_dim, f"实体维度 {E} 与 entity_dim {self.entity_dim} 不匹配"

        # Step 1: 实体投影
        h = self.entity_proj(entity_window)  # (T, N, gnn_hidden_dim)

        # Step 2: 帧内关系建模（GAT-lite：全连接自注意力）
        # 将 T 和 N 合并为 batch 维度
        h_flat = h.view(T * N, self.gnn_hidden_dim)  # (T*N, gnn_hidden_dim)

        q = self.q_proj(h_flat)
        k = self.k_proj(h_flat)
        v = self.v_proj(h_flat)

        # 多头注意力
        q = q.view(T * N, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, T*N, head_dim)
        k = k.view(T * N, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(T * N, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (num_heads, T*N, head_dim)

        attn_out = attn_out.transpose(0, 1).contiguous().view(T * N, self.gnn_hidden_dim)
        attn_out = self.out_proj(attn_out)  # (T*N, gnn_hidden_dim)

        # 残差连接
        h_flat = h_flat + attn_out
        h_flat = self.norm(h_flat)
        # (T*N, gnn_hidden_dim) → (T, N, gnn_hidden_dim)
        h = h_flat.view(T, N, self.gnn_hidden_dim)

        # Step 3: 跨帧时序整合
        # 聚合 N 个实体的关系为单向量
        h_frame = h.mean(dim=1)  # (T, gnn_hidden_dim)

        # 使用双向 GRU 捕捉跨帧因果关系
        gru_out, _ = self.temporal_encoder(h_frame.unsqueeze(1))  # (T, 1, gnn_hidden_dim*2)
        gru_out = gru_out.squeeze(1)  # (T, gnn_hidden_dim*2)

        relational_features = self.temporal_proj(gru_out)  # (T, gnn_hidden_dim)

        return relational_features  # (T, GNN_HIDDEN_DIM)
