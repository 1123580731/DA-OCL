"""
DA-OCL 逻辑约束层
在 SSM 输出的时序压缩表征上施加元规则软约束
使用多头注意力机制
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da_ocl.core.constants import RULE_DIM, SSM_STATE_DIM


class RuleConstraintLayer(nn.Module):
    """
    逻辑约束层：多头注意力机制。

    在 SSM 输出的时序压缩表征上施加元规则软约束。
    约束对象从单帧快照升级为一段经历（事件级别）。

    输入:
      - query: SSM 输出表征 h_t ∈ R^{SSM_STATE_DIM}
      - key/value: 元规则库 R ∈ R^{num_rules x RULE_DIM}

    输出: z_slow ∈ R^{RULE_DIM}，约束后的慢记忆表征

    Args:
        ssm_state_dim: SSM 隐状态维度（默认 512）
        rule_dim: 元规则向量维度（默认 256）
        num_rules: 元规则库中规则数量（默认 32）
        num_heads: 多头注意力的头数（默认 4）
    """

    def __init__(
        self,
        ssm_state_dim: int = SSM_STATE_DIM,
        rule_dim: int = RULE_DIM,
        num_rules: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.ssm_state_dim = ssm_state_dim
        self.rule_dim = rule_dim
        self.num_rules = num_rules
        self.num_heads = num_heads

        assert ssm_state_dim % num_heads == 0, (
            f"ssm_state_dim {ssm_state_dim} 必须能被 num_heads {num_heads} 整除"
        )

        self.head_dim = ssm_state_dim // num_heads

        # 可学习的元规则库（可训练参数）
        self.register_parameter(
            "rule_library",
            nn.Parameter(torch.randn(num_rules, rule_dim)),
        )

        # SSM 输出到 query 的投影
        self.q_proj = nn.Linear(ssm_state_dim, ssm_state_dim)
        # 元规则库的 key/value 投影
        self.k_proj = nn.Linear(rule_dim, ssm_state_dim)
        self.v_proj = nn.Linear(rule_dim, ssm_state_dim)

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=ssm_state_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(ssm_state_dim, rule_dim),
            nn.LayerNorm(rule_dim),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(ssm_state_dim)

    def forward(self, ssm_output: torch.Tensor) -> torch.Tensor:
        """
        对 SSM 输出施加元规则约束。

        Args:
            ssm_output: (SSM_STATE_DIM,) SSM 压缩后的表征

        Returns:
            rule_out: (RULE_DIM,) 元规则约束后的慢记忆表征
        """
        # 确保输入是 3D batched：(batch, 1, ssm_state_dim)
        squeeze = False
        if ssm_output.dim() == 1:
            ssm_output = ssm_output.unsqueeze(0)  # (1, ssm_state_dim)
            squeeze = True

        assert ssm_output.dim() == 2, f"ssm_output 应为 1D 或 2D，实际为 {ssm_output.dim()}"
        batch_size = ssm_output.shape[0]

        # 投影 SSM 输出为 query，并确保为 3D：(batch, 1, ssm_state_dim)
        query = self.q_proj(ssm_output)  # (batch, ssm_state_dim)
        query = query.unsqueeze(1)  # (batch, 1, ssm_state_dim) — seq_len=1

        # 投影元规则库为 key/value
        # rule_library: (num_rules, rule_dim)
        key = self.k_proj(self.rule_library)  # (num_rules, ssm_state_dim)
        value = self.v_proj(self.rule_library)  # (num_rules, rule_dim)

        # 扩展到 batch 维度：(batch, num_rules, ssm_state_dim)
        key = key.unsqueeze(0).expand(batch_size, -1, -1)
        value = value.unsqueeze(0).expand(batch_size, -1, -1)

        # 多头注意力（batch_first=True：输入 (batch, seq, embed)）
        attn_out, _ = self.multihead_attn(query, key, value)  # (batch, 1, ssm_state_dim)
        attn_out = attn_out.squeeze(1)  # (batch, ssm_state_dim)

        # 残差连接 + 归一化
        residual = query.squeeze(1)  # (batch, ssm_state_dim)
        out = self.norm(residual + attn_out)  # (batch, ssm_state_dim)

        # 输出投影到 RULE_DIM
        out = self.out_proj(out)  # (batch, rule_dim)

        if squeeze:
            out = out.squeeze(0)  # (rule_dim,)

        return out  # (RULE_DIM,) 或 (batch, RULE_DIM)

    def get_rule_library(self) -> torch.Tensor:
        """返回当前元规则库（用于分析）"""
        return self.rule_library
