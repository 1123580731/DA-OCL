"""
DA-OCL 梦境生成器
在规律边界采样，生成梦境经验用于强化慢记忆
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da_ocl.core.constants import (
    DREAM_NUM_SAMPLES,
    DREAM_VARIANCE_INIT,
    SLOW_MEMORY_DIM,
    SSM_STATE_DIM,
)


class DreamGenerator(nn.Module):
    """
    梦境生成器。

    在规律边界采样，生成梦境经验来测试和强化慢记忆的泛化能力。

    方法：
    1. 在慢记忆的表征空间中进行插值采样
    2. 在存储的因果链两端点之间采样
    3. 在规则约束的边界处采样
    """

    def __init__(
        self,
        num_samples: int = DREAM_NUM_SAMPLES,
        variance_init: float = DREAM_VARIANCE_INIT,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.variance = nn.Parameter(torch.tensor(variance_init))

    def sample_along_causal_chain(
        self,
        chain_start: torch.Tensor,
        chain_end: torch.Tensor,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """
        在因果链两端点之间均匀插值采样。

        Args:
            chain_start: (batch, dim) 链起点
            chain_end: (batch, dim) 链终点
            num_samples: 采样数量（默认 self.num_samples）

        Returns:
            (batch, num_samples, dim) 梦境样本
        """
        n = num_samples if num_samples is not None else self.num_samples
        t = torch.linspace(0, 1, n, device=chain_start.device, dtype=chain_start.dtype)
        t = t.view(1, -1, 1)  # (1, n, 1)
        return t * chain_end.unsqueeze(1) + (1 - t) * chain_start.unsqueeze(1)

    def sample_boundary_variations(
        self,
        center: torch.Tensor,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """
        在规律中心周围进行边界变化采样。

        添加方差尺度的噪声来探索泛化边界。

        Args:
            center: (batch, dim) 规律中心
            num_samples: 采样数量

        Returns:
            (batch, num_samples, dim) 梦境样本
        """
        n = num_samples if num_samples is not None else self.num_samples
        noise = torch.randn(center.shape[0], n, center.shape[1], device=center.device, dtype=center.dtype)
        # 噪声幅度受方差参数控制
        samples = center.unsqueeze(1) + self.variance * noise
        return samples

    def generate(
        self,
        slow_memory: nn.Module,
        chain_experiences: list[dict] | None = None,
        boundary_centers: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        生成梦境样本。

        Args:
            slow_memory: 慢记忆模型
            chain_experiences: 因果链经验列表
            boundary_centers: (batch, dim) 规律边界中心

        Returns:
            dream_samples: (total_samples, dim) 梦境样本
            dream_metadata: 辅助信息
        """
        device = next(slow_memory.parameters()).device
        dtype = next(slow_memory.parameters()).dtype

        dream_samples_list = []

        # 方式 1: 基于因果链插值
        if chain_experiences is not None and len(chain_experiences) >= 2:
            starts = []
            ends = []
            for i in range(len(chain_experiences) - 1):
                s = chain_experiences[i].get("embedding", None)
                e = chain_experiences[i + 1].get("embedding", None)
                if s is not None and e is not None:
                    starts.append(s)
                    ends.append(e)
            if starts:
                start_batch = torch.stack(starts, dim=0).to(device, dtype)
                end_batch = torch.stack(ends, dim=0).to(device, dtype)
                chain_samples = self.sample_along_causal_chain(start_batch, end_batch)
                dream_samples_list.append(chain_samples)

        # 方式 2: 基于边界变化
        if boundary_centers is not None:
            boundary_samples = self.sample_boundary_variations(boundary_centers)
            dream_samples_list.append(boundary_samples)

        # 方式 3: 随机采样（补充）
        if not dream_samples_list:
            total_dim = SLOW_MEMORY_DIM
            random_samples = torch.randn(
                1, self.num_samples, total_dim, device=device, dtype=dtype
            )
            dream_samples_list.append(random_samples)

        # 合并所有样本
        # 所有样本格式为 (num_chains, num_samples, dim)
        # 在 dim=0 上拼接（不同方法的 chain 数量可能不同）
        all_samples = torch.cat(dream_samples_list, dim=0)  # (total_chains, num_samples, dim)
        # 展平为 (total_chains * num_samples, dim)
        dream_samples = all_samples.view(-1, all_samples.shape[-1])

        return {
            "dream_samples": dream_samples,
            "num_samples": dream_samples.shape[0],
        }
