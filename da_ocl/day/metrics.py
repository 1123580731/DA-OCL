"""
DA-OCL 双指标计算
Imp_batch 和 Surp_batch 的计算
"""

from __future__ import annotations

import torch

from da_ocl.core.constants import (
    ALPHA_IMPORTANCE,
    COMPOSITE_DIM,
    DEATH_R_PENALTY_THRESHOLD,
    RULE_DIM,
    SSM_STATE_DIM,
    SURPRISE_SIGMA,
    WINDOW_SIZE_DEFAULT,
)


def compute_importance(
    coverage_rate: float,
    r_penalty: float,
    alpha: float = ALPHA_IMPORTANCE,
    death_threshold: float = DEATH_R_PENALTY_THRESHOLD,
) -> float:
    """
    计算重要性指标。

    Imp_t = alpha * (1 - CoverageRate) + (1 - alpha) * r_penalty

    注意：r_penalty 独立设有绝对阈值（DEATH_R_PENALTY_THRESHOLD），
    超过阈值的经验直接进入保护区，不参与此计算。

    Args:
        coverage_rate: 规律库覆盖率 ∈ [0, 1]
        r_penalty: 奖励惩罚值（仅死亡相关 > 0）
        alpha: (1-CoverageRate) 的权重
        death_threshold: 保护区触发阈值

    Returns:
        float: 重要性分数 ∈ [0, ∞)
    """
    # r_penalty 归一化（除以死亡阈值）
    normalized_r = min(r_penalty, death_threshold) / max(1.0, death_threshold)

    importance = alpha * (1.0 - coverage_rate) + (1.0 - alpha) * normalized_r
    return float(importance)


def compute_surprise(
    o_batch: torch.Tensor,
    o_hat_batch: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    计算窗口级别的惊喜度。

    Surp_batch = ||O_batch - O_hat_batch||_2

    Args:
        o_batch: (SSM_STATE_DIM,) 当前窗口慢记忆表征
        o_hat_batch: (SSM_STATE_DIM,) 检索到的规律表征

    Returns:
        (scalar, per_frame)
        - scalar: 窗口级别的惊喜度标量
        - per_frame: (T,) 每帧的惊喜度（用于联合掩码的 max 计算）
    """
    assert o_batch.shape[-1] == SSM_STATE_DIM
    assert o_hat_batch.shape[-1] == SSM_STATE_DIM

    diff = o_batch - o_hat_batch  # (SSM_STATE_DIM,)
    scalar = torch.norm(diff, p=2).item()

    # 简化：将等量惊喜度分配给每一帧
    T = WINDOW_SIZE_DEFAULT
    per_frame = torch.full((T,), scalar / T, device=o_batch.device)

    return scalar, per_frame


def is_death_experience(
    r_penalty: float,
    death_threshold: float = DEATH_R_PENALTY_THRESHOLD,
) -> bool:
    """
    判断是否为死亡经验。

    Args:
        r_penalty: 奖励惩罚值
        death_threshold: 触发保护区的绝对阈值

    Returns:
        True if r_penalty >= death_threshold
    """
    return r_penalty >= death_threshold


def compute_task_value(
    composite: torch.Tensor,
    rule_out: torch.Tensor,
    reward: float,
) -> float:
    """
    计算任务导向价值分数。

    用于任务导向区的排序。
    综合考虑：与目标规律的一致性 + 即时奖励。

    Args:
        composite: (COMPOSITE_DIM,) 复合特征
        rule_out: (RULE_DIM,) 元规则约束输出
        reward: 即时奖励

    Returns:
        float: 任务价值分数
    """
    # 表征质量分数（复合特征 L2 范数作为活跃度代理）
    activity_score = torch.norm(composite).item() / (COMPOSITE_DIM ** 0.5)

    # 规则一致性分数
    rule_score = torch.norm(rule_out).item() / (RULE_DIM ** 0.5)

    # 综合分数
    task_value = 0.4 * activity_score + 0.3 * rule_score + 0.3 * max(0, reward)

    return float(task_value)
