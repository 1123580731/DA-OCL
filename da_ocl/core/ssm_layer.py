"""
DA-OCL SSM 规律压缩层
在时间维度压缩时序信息，提取跨帧时序不变的物理规律表征
优先使用 Mamba-SSM，失败则自动降级到纯 PyTorch 实现
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from da_ocl.core.constants import GNN_HIDDEN_DIM, SSM_STATE_DIM


def _try_import_mamba() -> bool:
    """尝试导入 mamba-ssm，返回是否成功"""
    try:
        from mamba_ssm import Mamba  # noqa: F401
        return True
    except ImportError:
        return False


# 尝试导入 mamba-ssm，失败则使用降级实现
# 函数在 TYPE_CHECKING 分支之前定义，避免 Python 3.13 解析顺序问题
_MAMBA_AVAILABLE: bool = _try_import_mamba()


# ============================================================
# 降级实现（纯 PyTorch）
# ============================================================


class FallbackSSMLayer(nn.Module):
    """
    纯 PyTorch SSM 降级实现（不依赖 mamba-ssm）。

    使用双向 GRU 压缩时序，后接投影层和 LayerNorm。
    在没有 mamba-ssm 的环境中提供功能完整的替代方案。

    Args:
        input_dim: 输入特征维度
        state_dim: SSM 隐状态维度（输出维度）
    """

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # 双向 GRU 压缩时序
        self.gru = nn.GRU(
            input_dim,
            state_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(state_dim * 2, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, input_dim) 或 (batch, T, input_dim)

        Returns:
            compressed_state: (state_dim,) 或 (batch, state_dim)

        Raises:
            ValueError: 如果序列长度为 0
        """
        if x.dim() == 2:
            # (T, input_dim) → (1, T, input_dim)
            x = x.unsqueeze(0)
            squeeze_output = True
        elif x.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(f"输入维度应为 2 或 3，实际为 {x.dim()}")

        if x.shape[1] == 0:
            raise ValueError("序列长度 T 不能为 0")

        outputs, _ = self.gru(x)  # (batch, T, state_dim*2)
        # 使用最后时间步的输出（双向拼接）
        last_h = outputs[:, -1, :]  # (batch, state_dim*2)
        out = self.activation(self.proj(last_h))  # (batch, state_dim)
        out = self.norm(out)

        if squeeze_output:
            out = out.squeeze(0)  # (state_dim,)

        return out


# ============================================================
# 主 SSM 压缩层（优先使用 Mamba）
# ============================================================


class SSMCompressor(nn.Module):
    """
    SSM 规律压缩层。

    在 GNN 输出的时序维度压缩，提取跨帧时序不变的物理规律表征。
    优先使用 Mamba-SSM，若不可用则自动降级到 FallbackSSMLayer。

    输入: relational_features ∈ R^{T x GNN_HIDDEN_DIM}
    输出: compressed_state ∈ R^{SSM_STATE_DIM}

    压缩方式: 时间维度压缩（所有 T 帧 → 1 个状态向量）

    Args:
        gnn_hidden_dim: GNN 输出维度（默认 256）
        ssm_state_dim: SSM 隐状态维度（默认 512）
        use_mamba: 是否优先使用 mamba-ssm（默认 True，失败自动降级）
    """

    def __init__(
        self,
        gnn_hidden_dim: int = GNN_HIDDEN_DIM,
        ssm_state_dim: int = SSM_STATE_DIM,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.ssm_state_dim = ssm_state_dim
        self.use_mamba = False  # 将在下面确定

        if use_mamba and _MAMBA_AVAILABLE:
            try:
                from mamba_ssm import Mamba

                # Mamba 配置：d_model=输入维度，d_state=SSM状态维度
                self.ssm = Mamba(
                    d_model=gnn_hidden_dim,
                    d_state=ssm_state_dim,
                    expand=2,
                )
                self.use_mamba = True
                self._impl = "mamba"
            except Exception as e:
                warnings.warn(f"mamba-ssm 初始化失败: {e}，使用 PyTorch 降级实现")
                self.ssm = FallbackSSMLayer(gnn_hidden_dim, ssm_state_dim)
                self.use_mamba = False
                self._impl = "gru_fallback"
        else:
            self.ssm = FallbackSSMLayer(gnn_hidden_dim, ssm_state_dim)
            self.use_mamba = False
            self._impl = "gru_fallback"
            if use_mamba:
                warnings.warn(
                    "mamba-ssm 不可用（未安装或导入失败），"
                    "使用 PyTorch 降级实现（FallbackSSMLayer）"
                )

        # 投影层确保输出维度严格等于 SSM_STATE_DIM
        if self.use_mamba:
            # Mamba 输出维度等于 d_model（即 gnn_hidden_dim）
            self.proj = nn.Linear(gnn_hidden_dim, ssm_state_dim)
            self.norm = nn.LayerNorm(ssm_state_dim)
        else:
            # Fallback 直接输出 ssm_state_dim
            self.proj = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, relational_features: torch.Tensor) -> torch.Tensor:
        """
        对时序关系特征进行压缩。

        Args:
            relational_features: (T, GNN_HIDDEN_DIM)

        Returns:
            compressed_state: (SSM_STATE_DIM,) 跨帧时序压缩表征

        Raises:
            AssertionError: 如果 T=0（空序列）
        """
        T = relational_features.shape[0]
        if T <= 0:
            raise ValueError(f"时序特征序列不能为空（T={T}）")

        if self.use_mamba:
            # Mamba 接收 (batch, seq_len, d_model)
            x = relational_features.unsqueeze(0)  # (1, T, gnn_hidden_dim)
            ssm_out = self.ssm(x)  # (1, T, gnn_hidden_dim)
            # 取最后时间步
            final_state = ssm_out[:, -1, :]  # (1, gnn_hidden_dim)
            out = self.norm(self.proj(final_state)).squeeze(0)  # (SSM_STATE_DIM,)
        else:
            # Fallback SSM: 需要 (batch, T, input_dim)
            x = relational_features.unsqueeze(0)  # (1, T, gnn_hidden_dim)
            out = self.ssm(x).squeeze(0)  # (SSM_STATE_DIM,)

        assert out.shape[-1] == self.ssm_state_dim, (
            f"输出维度 {out.shape[-1]} 不等于 SSM_STATE_DIM {self.ssm_state_dim}"
        )
        return out

    @property
    def implementation(self) -> str:
        """返回当前使用的实现方式：'mamba' 或 'gru_fallback'"""
        return self._impl
