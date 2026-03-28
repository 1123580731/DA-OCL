"""
DA-OCL Regulation 模块
调节轴：重要性、惊喜度、时间衰减
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da_ocl.core.constants import (
    IMPORTANCE_DECAY_RATE,
    IMPORTANCE_MEAN_INIT,
    REGULATION_DIM,
    SLOW_MEMORY_DIM,
    SSM_STATE_DIM,
    SURPRISE_DECAY_RATE,
    SURPRISE_MEAN_INIT,
    TEMPORAL_DECAY_COEFFICIENT,
)


class ImportanceRegulation(nn.Module):
    """
    重要性调节轴。

    作用：根据日间累积的重要性信号，调节慢记忆参数的更新强度。

    实现：高重要性 → 低学习率（保护关键知识）
          低重要性 → 高学习率（允许探索）
    """

    def __init__(
        self,
        dim: int = SLOW_MEMORY_DIM,
        init_mean: float = IMPORTANCE_MEAN_INIT,
        decay_rate: float = IMPORTANCE_DECAY_RATE,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.decay_rate = decay_rate

        # 可学习的均值参数（logit 空间）
        self.mean_logit = nn.Parameter(torch.tensor(init_mean).log())

        # 可学习的衰减参数
        self.decay = nn.Parameter(torch.tensor(decay_rate))

    @property
    def mean(self) -> float:
        """获取当前均值（概率空间）"""
        return torch.sigmoid(self.mean_logit).item()

    def forward(self, importance_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            importance_signal: (batch, dim) 或 (dim,) 重要性信号

        Returns:
            调节后的更新强度 ∈ [0, 1]
        """
        # 指数衰减：高重要性 → 低强度
        strength = torch.exp(-self.decay * importance_signal.mean())
        return strength.clamp(0.0, 1.0)


class SurpriseRegulation(nn.Module):
    """
    惊喜度调节轴。

    作用：根据日间累积的惊喜度信号，控制慢记忆参数的更新范围。

    实现：高惊喜度 → 大更新（快速适应新模式）
          低惊喜度 → 小更新（维持稳定）
    """

    def __init__(
        self,
        dim: int = SLOW_MEMORY_DIM,
        init_mean: float = SURPRISE_MEAN_INIT,
        decay_rate: float = SURPRISE_DECAY_RATE,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.decay_rate = decay_rate

        # 可学习的均值参数（logit 空间）
        self.mean_logit = nn.Parameter(torch.tensor(init_mean).log())

        # 可学习的动态范围
        self.range_param = nn.Parameter(torch.tensor(1.0))

    @property
    def mean(self) -> float:
        """获取当前均值"""
        return torch.sigmoid(self.mean_logit).item()

    def forward(self, surprise_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            surprise_signal: (batch, dim) 或 (dim,) 惊喜度信号

        Returns:
            更新幅度缩放因子 ∈ [0, +∞)
        """
        # 惊喜度越高，更新幅度越大
        raw = torch.sigmoid(self.mean_logit) + self.range_param * surprise_signal.mean()
        return raw.clamp(min=0.0)


class TemporalDecayRegulation(nn.Module):
    """
    时间衰减调节轴。

    作用：根据参数的新鲜度，动态调节学习率。

    实现：新鲜参数（近期更新）→ 高学习率
          陈旧参数（长期未更新）→ 低学习率（防止遗忘）
    """

    def __init__(
        self,
        dim: int = SLOW_MEMORY_DIM,
        decay_coefficient: float = TEMPORAL_DECAY_COEFFICIENT,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.decay_coefficient = decay_coefficient

        # 可学习的衰减系数
        self.coeff = nn.Parameter(torch.tensor(decay_coefficient))

        # 参数新鲜度记录（由 MaskHistory 在日间累积）
        self._freshness: torch.Tensor | None = None

    def set_freshness(self, freshness: torch.Tensor) -> None:
        """
        设置参数新鲜度向量。

        Args:
            freshness: (dim,) 参数新鲜度 ∈ [0, 1]
        """
        self._freshness = freshness

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        """
        Args:
            param: (dim,) 慢记忆参数

        Returns:
            时间衰减后的学习率缩放因子 ∈ [0, 1]
        """
        if self._freshness is None:
            # 无新鲜度信息，均匀衰减
            return torch.ones_like(param) * 0.5

        # 新鲜度高 → 衰减慢 → 高学习率
        # 新鲜度低 → 衰减快 → 低学习率
        decay = torch.exp(-self.coeff * (1.0 - self._freshness))
        return decay.clamp(0.0, 1.0)


class RegulationMixer(nn.Module):
    """
    三轴调节混合器。

    将重要性、惊喜度、时间衰减三个调节轴合并为一个综合调节向量。

    FinalRegulation = ImportanceRegulation * SurpriseRegulation * TemporalDecayRegulation
    """

    def __init__(
        self,
        dim: int = SLOW_MEMORY_DIM,
        importance_init: float = IMPORTANCE_MEAN_INIT,
        surprise_init: float = SURPRISE_MEAN_INIT,
        temporal_coeff: float = TEMPORAL_DECAY_COEFFICIENT,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.importance_reg = ImportanceRegulation(dim, importance_init)
        self.surprise_reg = SurpriseRegulation(dim, surprise_init)
        self.temporal_reg = TemporalDecayRegulation(dim, temporal_coeff)

    def set_freshness(self, freshness: torch.Tensor) -> None:
        """设置新鲜度（由 MaskHistory 提供）"""
        self.temporal_reg.set_freshness(freshness)

    def forward(
        self,
        importance_signal: torch.Tensor,
        surprise_signal: torch.Tensor,
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            importance_signal: 重要性信号
            surprise_signal: 惊喜度信号
            param: 慢记忆参数

        Returns:
            综合调节后的更新强度 ∈ [0, 1]
        """
        imp_strength = self.importance_reg(importance_signal)
        surp_scale = self.surprise_reg(surprise_signal)
        temporal_scale = self.temporal_reg(param)

        # 三轴乘积混合，返回标量
        return (imp_strength * surp_scale * temporal_scale.mean()).clamp(0.0, 1.0)
