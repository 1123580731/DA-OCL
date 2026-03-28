"""
DA-OCL 快记忆系统
基于现代霍普菲尔德网络（MHN）的超线性容量联想工作台
存储因果窗口的联合表征，支持单步联想检索
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from da_ocl.core.constants import COMPOSITE_DIM, MHN_BETA, MHN_RETRIEVE_ITERS
from da_ocl.memory.fast_memory_ops import mhn_retrieve, store_patterns


class FastMemory(nn.Module):
    """
    快记忆：基于现代霍普菲尔德网络（MHN）。

    特性：
    - 超线性存储容量（MHN 理论保证，>1000 个窗口表征）
    - 单步联想检索
    - 存储整个因果窗口的联合表征（而非单帧表征）

    Args:
        capacity: 最大存储窗口数量（默认 5000）
        composite_dim: 复合特征维度（默认 1536）
    """

    def __init__(
        self,
        capacity: int = 5000,
        composite_dim: int = COMPOSITE_DIM,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.composite_dim = composite_dim

        # 存储槽：使用 buffer（不参与梯度计算）
        self.register_buffer("keys", torch.zeros(capacity, composite_dim))
        self.register_buffer("values", torch.zeros(capacity, composite_dim))
        self.register_buffer("usage_count", torch.zeros(capacity))  # LRU 使用计数
        self.register_buffer("_stored_count", torch.tensor(0))

        self._stored_count_local = 0

    @property
    def stored_count(self) -> int:
        """当前存储的窗口数量"""
        return int(self._stored_count_local)

    @property
    def is_empty(self) -> bool:
        return self._stored_count_local == 0

    def store(self, composite_features: torch.Tensor) -> None:
        """
        存储一个因果窗口的复合特征。

        支持两种模式：
        1. 单个复合特征：composite_features ∈ (composite_dim,)
           → 直接存储
        2. 多帧表征列表：composite_features ∈ (num_frames, composite_dim)
           → 存储每帧表征的聚合（平均池化）

        Args:
            composite_features: 复合特征张量
        """
        if composite_features.dim() == 1:
            new_key = composite_features.detach()
            new_value = composite_features.detach()
        elif composite_features.dim() == 2:
            # 多帧聚合：平均池化
            new_key = composite_features.mean(dim=0).detach()
            new_value = composite_features.detach()  # 保留原始多帧表征
        else:
            raise ValueError(
                f"composite_features 应为 1D 或 2D，实际为 {composite_features.dim()}D"
            )

        current_keys = self.keys[:self._stored_count_local]
        current_values = self.values[:self._stored_count_local]

        updated_keys, updated_values = store_patterns(
            current_keys,
            current_values,
            new_key,
            new_value,
            self.capacity,
        )

        # 更新 buffer
        num_to_update = min(updated_keys.shape[0], self.capacity)
        self.keys[:num_to_update] = updated_keys[:num_to_update]
        self.values[:num_to_update] = updated_values[:num_to_update]

        # 更新计数：若之前未满则递增；若已满则保持 capacity（FIFO 替换后实际存储数不变）
        if self._stored_count_local < self.capacity:
            self._stored_count_local = min(self._stored_count_local + 1, self.capacity)
        # 若已满（capacity），FIFO 替换后仍然是 capacity（替换不改变数量）

        # 更新 LRU 计数（提升被检索过的）
        self._update_usage()

    def _update_usage(self) -> None:
        """所有条目的使用计数 +1（用于 LRU）"""
        count = int(self._stored_count_local)
        if count > 0:
            self.usage_count[:count] += 1

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        联想检索。

        Args:
            query: (composite_dim,) 查询向量

        Returns:
            (best_match, attention_weights)
            - best_match: (composite_dim,) 最相似的存储表征
            - attention_weights: (stored_count,) 每个存储表征的注意力权重

        Raises:
            RuntimeError: 未存储任何内容时调用
        """
        if self._stored_count_local == 0:
            raise RuntimeError("快记忆为空，无法执行检索（请先调用 store）")

        stored_keys = self.keys[:self._stored_count_local]
        stored_values = self.values[:self._stored_count_local]

        best_key, best_value, attention_weights = mhn_retrieve(
            query=query,
            keys=stored_keys,
            values=stored_values,
            beta=MHN_BETA,
            max_iters=MHN_RETRIEVE_ITERS,
        )

        # 更新 LRU：将检索命中的条目使用计数清零
        best_idx = attention_weights.argmax().item()
        if best_idx < self._stored_count_local:
            self.usage_count[best_idx] = 0.0

        return best_value, attention_weights

    def get_retrieval_distribution(
        self, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取检索分布（软检索，而非硬索引）。

        用于夜晚蒸馏损失计算。

        Args:
            query: (composite_dim,) 查询向量

        Returns:
            (keys, probs) - 所有存储键和对应的概率分布
        """
        if self._stored_count_local == 0:
            raise RuntimeError("快记忆为空")

        stored_keys = self.keys[:self._stored_count_local]
        stored_values = self.values[:self._stored_count_local]

        # 计算余弦相似度分布
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        keys_norm = F.normalize(stored_keys, dim=-1)
        similarities = F.cosine_similarity(query_norm, keys_norm, dim=-1)
        probs = F.softmax(MHN_BETA * similarities, dim=0)

        # 加权检索值
        best_match = (probs.unsqueeze(-1) * stored_values).sum(dim=0)

        return best_match, probs

    def get_state_dict(self) -> dict:
        """返回存储状态（用于 checkpoint）"""
        return {
            "keys": self.keys[: self._stored_count_local].clone(),
            "values": self.values[: self._stored_count_local].clone(),
            "usage_count": self.usage_count[: self._stored_count_local].clone(),
            "_stored_count_local": self._stored_count_local,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 checkpoint 恢复存储状态"""
        keys = state["keys"]
        values = state["values"]
        count = state["_stored_count_local"]

        self._stored_count_local = count
        self.keys[:count] = keys
        self.values[:count] = values
        if "usage_count" in state:
            self.usage_count[:count] = state["usage_count"]
