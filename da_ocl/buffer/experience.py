"""
DA-OCL Experience 数据类
统一经验格式
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from da_ocl.core.constants import COMPOSITE_DIM, SSM_STATE_DIM, WINDOW_SIZE_DEFAULT


@dataclass
class Experience:
    """
    Buffer 中存储的最小单位。

    存储一个完整因果窗口的经验，包括：
    - 原始窗口数据（用于慢记忆再处理）
    - 复合特征（用于快记忆和蒸馏）
    - 指标（用于优先级排序和替换策略）
    """
    # === 核心数据 ===
    entity_window: torch.Tensor  # (T, entity_dim) 因果窗口的实体嵌入
    composite: torch.Tensor      # (COMPOSITE_DIM,) 复合特征
    ssm_output: torch.Tensor    # (SSM_STATE_DIM,) SSM 输出（用于蒸馏）

    # === 指标 ===
    surprise: float             # 惊喜度
    importance: float           # 重要性
    coverage_rate: float       # 规律覆盖情况
    r_penalty: float           # 奖励惩罚值（仅死亡相关 > 0）

    # === 死亡与因果链 ===
    is_death: bool = False     # 是否为死亡经验
    causal_chain: list[int] | None = None  # 因果链追溯的帧索引

    # === 元数据 ===
    timestep: int = 0         # 记录时的全局时间步
    priority_score: float = 0.0  # 综合优先级分数
    pattern_overlap: float = 0.0  # 与规律库的重叠度（多样性区使用）
    task_value: float = 0.0      # 任务价值分数（任务导向区使用）

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 checkpoint）"""
        return {
            "entity_window": self.entity_window.cpu(),
            "composite": self.composite.cpu(),
            "ssm_output": self.ssm_output.cpu(),
            "surprise": self.surprise,
            "importance": self.importance,
            "coverage_rate": self.coverage_rate,
            "r_penalty": self.r_penalty,
            "is_death": self.is_death,
            "causal_chain": self.causal_chain,
            "timestep": self.timestep,
            "priority_score": self.priority_score,
            "pattern_overlap": self.pattern_overlap,
            "task_value": self.task_value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Experience":
        """从字典反序列化"""
        return cls(
            entity_window=d["entity_window"],
            composite=d["composite"],
            ssm_output=d["ssm_output"],
            surprise=d["surprise"],
            importance=d["importance"],
            coverage_rate=d["coverage_rate"],
            r_penalty=d["r_penalty"],
            is_death=d.get("is_death", False),
            causal_chain=d.get("causal_chain"),
            timestep=d.get("timestep", 0),
            priority_score=d.get("priority_score", 0.0),
            pattern_overlap=d.get("pattern_overlap", 0.0),
            task_value=d.get("task_value", 0.0),
        )
