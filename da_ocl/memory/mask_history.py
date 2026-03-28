"""
DA-OCL 掩码历史累积管理器
记录每日慢记忆参数的掩码强度，供夜晚激活索引使用
掩码历史 = 逐参数累积 max
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class MaskHistory:
    """
    掩码历史累积器。

    白天阶段记录每个参数的联合掩码强度（逐参数取 max）。
    夜晚阶段使用掩码历史作为激活范围的索引——只对有过沉默写入记录的参数做激活强化。

    关键语义：
    - 掩码历史记录的是「哪些参数在白天被标记为需要写入」
    - 掩码值越大 → 该参数对应的规律越需要强化
    - 掩码值接近 0 → 该参数从未被激活，不参与夜晚更新

    使用方式：
    1. 白天：mask_history.accumulate(param_name, mask_value)
    2. 夜晚：activated_params = mask_history.get_activated_params(threshold=0.01)
    3. 整合完成后：mask_history.clear()
    """

    def __init__(self) -> None:
        self._history: Dict[str, float] = {}  # param_name → max mask value
        self._cumulative_count: Dict[str, int] = {}  # param_name → 累积次数
        self._surprise_history: list[float] = []  # 惊喜度滑动历史（独立追踪）
        self._surprise_window_size: int = 20

    def accumulate(self, param_name: str, mask_value: float) -> None:
        """
        累积掩码值（逐参数取 max）。

        Args:
            param_name: 参数名称
            mask_value: 联合掩码值 ∈ [0, 1]
        """
        if mask_value <= 0.0:
            return  # 零掩码不记录

        current_max = self._history.get(param_name, 0.0)
        self._history[param_name] = max(current_max, mask_value)

        count = self._cumulative_count.get(param_name, 0)
        self._cumulative_count[param_name] = count + 1

    def get_accumulated(self) -> dict[str, float]:
        """
        获取所有累积的掩码值。

        Returns:
            {param_name: accumulated_mask_value}
        """
        return self._history.copy()

    def get_density(self) -> float:
        """
        获取平均掩码强度（所有已记录参数掩码值的均值）。

        与 activation_density 不同，这里返回的是「掩码有多强」，而非「有多少参数被激活」。

        Returns:
            float ∈ [0, 1] — 所有记录掩码的平均值
        """
        return self.activation_density()

    def get_avg_surprise(self) -> float:
        """
        获取滑动窗口内的平均惊喜度。

        惊喜度由白天 observe 流程中的 SurpriseDetector 独立追踪，
        存储在 _surprise_history 中，与掩码强度无关。

        Returns:
            float — 滑动窗口内所有惊喜度的均值（若历史为空则为 0.0）
        """
        if not self._surprise_history:
            return 0.0
        return sum(self._surprise_history) / len(self._surprise_history)

    def record_surprise(self, surprise: float) -> None:
        """
        记录一个惊喜度值（白天流程调用）。

        Args:
            surprise: 当前窗口的惊喜度标量
        """
        self._surprise_history.append(float(surprise))
        if len(self._surprise_history) > self._surprise_window_size:
            self._surprise_history = self._surprise_history[-self._surprise_window_size:]

    def reset(self) -> None:
        """重置掩码历史（alias for clear）"""
        self.clear()

    @staticmethod
    def accumulate_from_model(model: nn.Module, mask_value: float) -> "MaskHistory":
        """
        从模型累积所有参数的掩码。

        Args:
            model: 慢记忆模型
            mask_value: 联合掩码值

        Returns:
            新的 MaskHistory 实例（不修改原实例）
        """
        history = MaskHistory()
        for name, _ in model.named_parameters():
            history.accumulate(name, mask_value)
        return history

    def get_mask(self, param_name: str) -> float:
        """
        查询特定参数的掩码值。

        Returns:
            mask value，或 0.0（从未记录）
        """
        return self._history.get(param_name, 0.0)

    def get_activated_params(
        self,
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        获取所有被激活的参数及其掩码值。

        Args:
            threshold: 激活阈值，低于此值的参数被忽略

        Returns:
            dict: {param_name: mask_value}
        """
        return {
            name: mask
            for name, mask in self._history.items()
            if mask >= threshold
        }

    def activation_density(self) -> float:
        """
        计算平均掩码强度（所有已记录掩码值的均值）。

        这是掩码「有多强」的度量，而非激活参数「有多少」的度量。
        当所有参数都积累了较强的掩码时密度才高。

        Returns:
            float ∈ [0, 1] — 所有记录掩码值的均值，若无记录则为 0.0
        """
        if not self._history:
            return 0.0
        values = list(self._history.values())
        return sum(values) / len(values)

    @property
    def total_activated(self) -> int:
        """被激活的参数数量"""
        return len(self._history)

    def merge(self, other: "MaskHistory") -> None:
        """
        合并另一个 MaskHistory（取逐参数 max）。

        用于多智能体场景或跨天累积。
        """
        for name, mask in other._history.items():
            current = self._history.get(name, 0.0)
            self._history[name] = max(current, mask)

            count = other._cumulative_count.get(name, 0)
            self._cumulative_count[name] = self._cumulative_count.get(name, 0) + count

    def clear(self) -> None:
        """清空掩码历史（夜晚整合完成后调用）"""
        self._history.clear()
        self._cumulative_count.clear()
        self._surprise_history.clear()

    def to_tensor(self) -> torch.Tensor:
        """
        将掩码历史转换为张量（用于 checkpoint 保存）。

        Returns:
            (N, 2) 张量：N=参数数量，[name_idx, mask_value]
        """
        items = list(self._history.items())
        if not items:
            return torch.zeros(0, 2)
        names, values = zip(*items)
        # 使用字符串哈希作为 name 的索引
        name_indices = torch.tensor([hash(n) % 1000000 for n in names], dtype=torch.long)
        return torch.stack([name_indices.float(), torch.tensor(values)], dim=-1)

    def summary(self) -> dict:
        """返回掩码历史的摘要统计"""
        if not self._history:
            return {
                "count": 0,
                "max": 0.0,
                "mean": 0.0,
                "density": 0.0,
                "avg_surprise": sum(self._surprise_history) / max(1, len(self._surprise_history)),
            }

        values = list(self._history.values())
        return {
            "count": len(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "density": self.activation_density(),
            "avg_surprise": sum(self._surprise_history) / max(1, len(self._surprise_history)),
        }
