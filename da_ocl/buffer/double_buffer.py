"""
DA-OCL 双缓冲管理器
解决白天检索和夜晚更新的并发安全问题
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict

import torch.nn as nn

from da_ocl.memory.slow_memory import SlowMemory

if TYPE_CHECKING:
    pass


class DoubleBufferManager:
    """
    双缓冲：解决白天检索和夜晚更新并发问题。

    - Buffer A: 当前服务白天检索的稳定版本
    - Buffer B: 夜晚正在被激活更新的版本

    切换策略：原子替换（整合完成且验证通过后）
    内存代价：慢记忆参数量 × 2

    掩码历史清零时机：整合完成且一致性验证通过后才清零。
    若触发回滚，掩码历史保留，下一次整合继续在此基础上累积。
    """

    def __init__(self, slow_memory: SlowMemory) -> None:
        self._slow_memory = slow_memory
        self._buffer_A: Dict[str, nn.Parameter] = {}
        self._buffer_B: Dict[str, nn.Parameter] | None = None
        self._is_night_active: bool = False

        # 初始化 Buffer A（当前慢记忆的快照）
        self._sync_buffer_A()

    def _sync_buffer_A(self) -> None:
        """将慢记忆的当前状态同步到 Buffer A"""
        self._buffer_A = {
            name: param.clone()
            for name, param in self._slow_memory.named_parameters()
        }

    def get_active_state(self) -> Dict[str, nn.Parameter]:
        """
        获取当前服务白天的稳定版本。

        夜晚激活期间持续返回 buffer_A。
        """
        return self._buffer_A

    def get_night_working_copy(self) -> SlowMemory:
        """
        获取夜晚更新的工作副本（SlowMemory 对象）。

        Returns:
            慢记忆对象的副本（用于夜晚梯度更新）
        """
        if not self._is_night_active:
            self.start_night_update()

        # 创建一个新的 SlowMemory 并加载 Buffer B 的状态
        night_copy = SlowMemory(use_mamba=False)
        night_copy.load_state_dict(self._buffer_B)

        # 将其参数设置为可训练（用于梯度更新）
        for param in night_copy.parameters():
            param.requires_grad = True

        return night_copy

    def start_night_update(self) -> None:
        """
        开始夜晚更新：复制 A 到 B。

        之后白天检索继续使用 A，夜晚更新操作 B。
        """
        if self._is_night_active:
            raise RuntimeError("夜晚更新已在进行中")

        self._buffer_B = {
            name: param.clone()
            for name, param in self._buffer_A.items()
        }
        self._is_night_active = True

    def commit(self, updated_slow_memory: SlowMemory) -> None:
        """
        夜晚整合完成且验证通过：原子替换 A ← B。

        更新慢记忆对象到最新参数。
        """
        if not self._is_night_active or self._buffer_B is None:
            raise RuntimeError("未开始夜晚更新，无法提交")

        # 将 B 的参数更新为夜晚整合后的值
        self._buffer_B = {
            name: param.clone()
            for name, param in updated_slow_memory.named_parameters()
        }

        # 原子替换 A ← B
        self._buffer_A = self._buffer_B
        self._buffer_B = None
        self._is_night_active = False

        # 将 A 的状态加载到实际的慢记忆对象
        self._slow_memory.load_state_dict(self._buffer_A)

    def rollback(self) -> None:
        """
        夜晚整合失败：丢弃 B，保留 A。

        掩码历史保留，下一次整合继续累积。
        """
        self._buffer_B = None
        self._is_night_active = False
        # A 保持不变，无需额外操作

    @property
    def is_night_active(self) -> bool:
        return self._is_night_active

    def apply_state_to_slow_memory(self) -> None:
        """
        将 Buffer A 的当前状态应用到慢记忆对象。

        用于初始化或恢复。
        """
        self._slow_memory.load_state_dict(self._buffer_A)
