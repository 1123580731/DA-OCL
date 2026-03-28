"""
DA-OCL 规律覆盖追踪器
追踪慢记忆对各因果规律的覆盖程度，用于计算联合掩码的第二个因子
"""

from __future__ import annotations

from typing import Dict


class CoverageTracker:
    """
    规律覆盖追踪器。

    追踪慢记忆对各因果规律的覆盖程度。
    用于计算 CoverageRate_batch（联合掩码的第二个因子 = 1 - global_mean_coverage）。

    语义：规律库越空（覆盖率低）→ 掩码值越大（越开放写入）
          规律库越满（覆盖率高）→ 掩码值越小（抑制已知规律的重复强化）
    """

    def __init__(self, initial_coverage: float = 0.0) -> None:
        """
        Args:
            initial_coverage: 初始覆盖率（默认 0.0，表示规律库完全空）
        """
        self._coverage_map: Dict[str, float] = {}  # rule_id → coverage ∈ [0,1]
        self._activation_count: Dict[str, int] = {}  # rule_id → 激活次数
        self._total_updates: int = 0
        self._global_mean: float = initial_coverage

    def update(self, activated_rules: list[str], batch_surprise: float) -> None:
        """
        根据当前窗口激活的规则更新覆盖图。

        惊喜度越高 → 该规律域的覆盖强度更新越大
        （发现新规律或强烈违反现有规律的区域需要提升覆盖）

        Args:
            activated_rules: 当前窗口激活的规则 ID 列表
            batch_surprise: 当前窗口的惊喜度（用于调整学习率）
        """
        self._total_updates += 1

        if not activated_rules:
            # 无激活规则时，随机探索规律库（增加整体覆盖率的不确定性）
            return

        # 学习率与惊喜度成正比：高惊喜 → 快速学习新规律
        lr = min(1.0, 0.1 + 0.1 * batch_surprise)

        for rule_id in activated_rules:
            if rule_id not in self._coverage_map:
                self._coverage_map[rule_id] = 0.0
                self._activation_count[rule_id] = 0

            self._activation_count[rule_id] += 1
            current = self._coverage_map[rule_id]
            # 覆盖率增量与惊喜度相关
            delta = lr * min(batch_surprise * 0.1, 1.0)
            self._coverage_map[rule_id] = min(1.0, current + delta)

        self._recompute_global_mean()

    def _recompute_global_mean(self) -> None:
        """重新计算全局平均覆盖率"""
        if not self._coverage_map:
            self._global_mean = 0.0
        else:
            self._global_mean = sum(self._coverage_map.values()) / len(self._coverage_map)

    def get_batch_coverage_rate(self) -> float:
        """
        返回当前窗口的规律覆盖缺口。

        即：1 - global_mean_coverage
        覆盖缺口越大 → 联合掩码的 (1 - CoverageRate) 因子越大 → 越开放写入
        """
        return max(0.0, 1.0 - self._global_mean)

    def get_global_mean(self) -> float:
        """返回全局平均覆盖率"""
        return self._global_mean

    def get_coverage_map(self) -> Dict[str, float]:
        """返回完整的覆盖图（用于分析）"""
        return self._coverage_map.copy()

    def get_coverage_for_rule(self, rule_id: str) -> float:
        """查询特定规则的覆盖率"""
        return self._coverage_map.get(rule_id, 0.0)

    def reset(self) -> None:
        """重置覆盖追踪状态"""
        self._coverage_map.clear()
        self._activation_count.clear()
        self._total_updates = 0
        self._global_mean = 0.0
