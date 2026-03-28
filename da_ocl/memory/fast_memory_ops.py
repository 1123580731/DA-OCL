"""
DA-OCL MHN 快记忆核心操作
现代霍普菲尔德网络（Modern Hopfield Network）的核心操作
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from da_ocl.core.constants import COMPOSITE_DIM, MHN_BETA, MHN_RETRIEVE_ITERS


def compute_energy(query: torch.Tensor, keys: torch.Tensor, beta: float = MHN_BETA) -> torch.Tensor:
    """
    计算 MHN 能量函数。

    E = -beta * Σ_i cos(query, key_i)
    能量越低 → query 与存储模式越相似

    Args:
        query: (composite_dim,) 查询向量
        keys: (num_stored, composite_dim) 存储的键向量
        beta: 能量缩放系数

    Returns:
        energy: (num_stored,) 每个存储键的能量值
    """
    # 归一化
    query_norm = F.normalize(query.unsqueeze(0), dim=-1)  # (1, dim)
    keys_norm = F.normalize(keys, dim=-1)  # (num_stored, dim)

    # 余弦相似度
    cos_sim = F.cosine_similarity(query_norm, keys_norm, dim=-1)  # (num_stored,)
    energy = -beta * cos_sim
    return energy


def mhn_retrieve(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor | None = None,
    beta: float = MHN_BETA,
    max_iters: int = MHN_RETRIEVE_ITERS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MHN 联想检索（连续更新算法）。

    通过能量最小化找到与 query 最相似的存储模式。

    Args:
        query: (composite_dim,) 查询向量
        keys: (num_stored, composite_dim) 存储的键向量
        values: (num_stored, composite_dim) 存储的值向量（与键相同则可省略）
        beta: 能量温度参数
        max_iters: 最大迭代次数

    Returns:
        best_key: (composite_dim,) 最相似的键向量
        best_value: (composite_dim,) 最相似的值向量
        attention_weights: (num_stored,) 每个存储模式的注意力权重
    """
    num_stored = keys.shape[0]
    device = keys.device

    # 初始化：query 的归一化版本
    retrieval_state = F.normalize(query, dim=-1).clone()  # (dim,)

    # 连续更新（类似 Hopfield 网络的能量下降）
    for _ in range(max_iters):
        # 计算与所有存储模式的相似度
        keys_norm = F.normalize(keys, dim=-1)
        retrieval_norm = F.normalize(retrieval_state.unsqueeze(0), dim=-1)  # (1, dim)
        similarities = F.cosine_similarity(retrieval_norm, keys_norm, dim=-1)  # (num_stored,)

        # softmax 注意力权重
        attention = F.softmax(beta * similarities, dim=0)  # (num_stored,)

        # 更新检索状态：所有存储键的加权平均
        new_state = (attention.unsqueeze(-1) * keys).sum(dim=0)  # (dim,)
        new_state = F.normalize(new_state, dim=-1)

        # 动量更新（防止震荡）
        retrieval_state = 0.5 * retrieval_state + 0.5 * new_state
        retrieval_state = F.normalize(retrieval_state, dim=-1)

    # 最终注意力权重
    keys_norm = F.normalize(keys, dim=-1)
    retrieval_norm = F.normalize(retrieval_state.unsqueeze(0), dim=-1)
    similarities = F.cosine_similarity(retrieval_norm, keys_norm, dim=-1)
    attention_weights = F.softmax(beta * similarities, dim=0)  # (num_stored,)

    # 找最佳匹配
    best_idx = attention_weights.argmax().item()
    best_key = keys[best_idx]
    best_value = values[best_idx] if values is not None else best_key

    return best_key, best_value, attention_weights


def store_patterns(
    keys: torch.Tensor,
    values: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在固定容量下存储新模式（FIFO 替换）。

    Args:
        keys: (N, composite_dim) 当前存储的键（N <= capacity）
        values: (N, composite_dim) 当前存储的值
        new_key: (composite_dim,) 新键向量
        new_value: (composite_dim,) 新值向量
        capacity: 最大容量

    Returns:
        (updated_keys, updated_values)
    """
    if keys.shape[0] == 0:
        # 首次存储
        return new_key.unsqueeze(0), new_value.unsqueeze(0)

    if keys.shape[0] < capacity:
        # 容量未满：追加
        updated_keys = torch.cat([keys, new_key.unsqueeze(0)], dim=0)
        updated_values = torch.cat([values, new_value.unsqueeze(0)], dim=0)
    else:
        # 容量已满：替换 LRU 最老的（简化：替换第一个）
        # 完整实现应使用 LRU 策略
        updated_keys = torch.cat([keys[1:], new_key.unsqueeze(0)], dim=0)
        updated_values = torch.cat([values[1:], new_value.unsqueeze(0)], dim=0)

    return updated_keys, updated_values
