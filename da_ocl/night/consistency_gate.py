"""
DA-OCL 一致性验证门控
验证夜间更新与日间因果结构的一致性
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da_ocl.core.constants import (
    CONSISTENCY_GATE_THRESHOLD,
    SLOW_MEMORY_DIM,
)


class ConsistencyGate(nn.Module):
    """
    一致性验证门控。

    作用：验证夜间更新是否与日间累积的因果结构一致，
    拒绝不一致的更新。

    机制：
    1. 计算更新前后因果结构的变化
    2. 与日间记录的一致性阈值比较
    3. 超过阈值则拒绝更新并回滚
    """

    def __init__(
        self,
        dim: int = SLOW_MEMORY_DIM,
        threshold: float = CONSISTENCY_GATE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.threshold = threshold

        # 可学习的一致性评估器
        self.evaluator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        # 记录上一致性状态
        self._last_consistency: float = 1.0

    def evaluate(
        self,
        before_state: dict[str, torch.Tensor],
        after_state: dict[str, torch.Tensor],
    ) -> tuple[bool, float]:
        """
        评估更新前后的一致性。

        Args:
            before_state: 更新前参数状态
            after_state: 更新后参数状态

        Returns:
            (accepted, consistency_score)
        """
        consistency_scores = []

        for name in before_state:
            if name not in after_state:
                continue

            before = before_state[name]
            after = after_state[name]

            # 计算参数变化的相对幅度
            diff = torch.norm(after - before, p=2) / (torch.norm(before, p=2) + 1e-8)

            # 评估一致性（使用神经网络评估器）
            combined = torch.cat([before.flatten(), after.flatten()])
            if combined.shape[0] > self.dim * 2:
                # 截断过长的参数向量
                combined = combined[: self.dim * 2]
            elif combined.shape[0] < self.dim * 2:
                # 填充
                combined = torch.nn.functional.pad(combined, (0, self.dim * 2 - combined.shape[0]))

            score = self.evaluator(combined.unsqueeze(0)).item()
            consistency_scores.append(score)

        # 平均一致性
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
        self._last_consistency = avg_consistency

        accepted = avg_consistency >= self.threshold
        return accepted, avg_consistency

    def validate_and_rollback(
        self,
        model: nn.Module,
        before_state: dict[str, torch.Tensor],
        after_state: dict[str, torch.Tensor],
    ) -> bool:
        """
        验证一致性，失败则回滚。

        Args:
            model: 慢记忆模型
            before_state: 更新前状态
            after_state: 更新后状态

        Returns:
            是否通过验证（通过=True，回滚后=False）
        """
        accepted, score = self.evaluate(before_state, after_state)

        if not accepted:
            # 回滚到更新前状态
            for name, param in model.named_parameters():
                if name in before_state:
                    param.data.copy_(before_state[name])
            return False
        return True

    def forward(
        self,
        before_state: dict[str, torch.Tensor],
        after_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        前向计算一致性损失（用于辅助训练）。

        Args:
            before_state: 更新前参数
            after_state: 更新后参数

        Returns:
            一致性损失（低一致性 → 高损失）
        """
        accepted, score = self.evaluate(before_state, after_state)
        # 返回负分数作为损失（鼓励高一致性）
        return torch.tensor(-score, device=next(iter(before_state.values())).device)
