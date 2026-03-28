"""
DA-OCL 因果窗口管理器
维护固定长度的 FIFO 窗口（5~8帧），输出因果窗口级别的实体表征序列
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from da_ocl.core.constants import ENTITY_DIM, WINDOW_SIZE_DEFAULT, WINDOW_SIZE_MAX, WINDOW_SIZE_MIN


class CausalWindowManager(nn.Module):
    """
    因果窗口管理器。

    接收连续帧的编码后实体特征，维护固定长度的 FIFO 窗口（长度 5~8，可配置）。
    输出 `EntityWindow = List[Tensor]`，长度为 T（因果窗口长度）。
    通知 BoundaryDetector 当前 batch 是否已触发新情景。

    关键认知语义：窗口长度不是工程批处理参数，而是认知窗口的长度——模型一次能看多长的因果链。

    Args:
        window_size: 因果窗口长度（5~8，认知有效区间，默认 6）
        device: 张量所在设备
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE_DEFAULT,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if not (WINDOW_SIZE_MIN <= window_size <= WINDOW_SIZE_MAX):
            raise ValueError(
                f"window_size {window_size} 超出认知有效区间 [{WINDOW_SIZE_MIN}, {WINDOW_SIZE_MAX}]"
            )

        self.window_size = window_size
        self._device = device or torch.device("cpu")
        # FIFO 缓冲区：存储最近的 window_size 个实体向量
        self._buffer: List[torch.Tensor] = []
        # 步数计数器
        self._step_count: int = 0

    @property
    def device(self) -> torch.device:
        return self._device

    def push(self, entity: torch.Tensor) -> List[torch.Tensor]:
        """
        推送一帧到窗口。

        Args:
            entity: (entity_dim,) 单帧编码后实体

        Returns:
            EntityWindow: 当前完整窗口（固定长度 T）
                         如果 buffer 尚未填满，返回当前所有帧（< T）
        """
        if entity.device != self._device:
            entity = entity.to(self._device)

        # 验证维度
        if entity.dim() != 1:
            raise ValueError(f"entity 应为 1D 张量，实际为 {entity.dim()}D")
        if entity.shape[-1] != ENTITY_DIM:
            raise ValueError(
                f"entity 维度 {entity.shape[-1]} 与 ENTITY_DIM {ENTITY_DIM} 不匹配"
            )

        # FIFO 入队
        self._buffer.append(entity.detach().clone())
        self._step_count += 1

        # 超出窗口长度时出队
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        return self._buffer.copy()

    def get_batch(self) -> Optional[torch.Tensor]:
        """
        返回当前窗口张量。

        Returns:
            (window_size, entity_dim) 或 None（窗口未满时返回 None）
        """
        if len(self._buffer) < self.window_size:
            return None  # 预热期：窗口尚未填满

        return torch.stack(self._buffer, dim=0)  # (T, entity_dim)

    def is_full(self) -> bool:
        """窗口是否已填满（可以输出完整 batch）"""
        return len(self._buffer) >= self.window_size

    @property
    def filled_frames(self) -> int:
        """已填充的帧数（0 ~ window_size）"""
        return len(self._buffer)

    @property
    def step_count(self) -> int:
        """全局步数计数器"""
        return self._step_count

    def reset(self) -> None:
        """重置窗口缓冲区和步数计数器"""
        self._buffer.clear()
        self._step_count = 0

    def to(self, device: torch.device) -> "CausalWindowManager":
        """移动到指定设备"""
        self._device = device
        self._buffer = [t.to(device) for t in self._buffer]
        return self
