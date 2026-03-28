"""
DA-OCL 环境接口基类
定义与 DA-OCL 智能体交互的环境接口契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EnvOutput:
    """环境输出数据结构"""
    obs: torch.Tensor  # (obs_dim,) 当前帧观测
    reward: float       # 奖励信号
    done: bool          # episode 是否结束
    info: dict[str, Any]  # 额外信息（可包含 r_penalty 等）


class BaseEnv(ABC):
    """
    环境基类。

    所有环境实现必须遵循此接口契约。
    DA-OCL 智能体通过此接口与环境交互。
    """

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """观测向量维度"""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """动作空间大小（离散动作）"""
        ...

    @property
    @abstractmethod
    def num_entities(self) -> int:
        """环境中实体的最大数量（用于 GNN 编码）"""
        ...

    @abstractmethod
    def reset(self) -> EnvOutput:
        """
        重置环境，返回初始观测。

        Returns:
            EnvOutput：包含初始 obs、reward=0.0、done=False

        Raises:
            RuntimeError: 环境内部状态错误
        """
        ...

    @abstractmethod
    def step(self, action: int) -> EnvOutput:
        """
        执行一个动作。

        Args:
            action: 动作索引 [0, action_dim)

        Returns:
            EnvOutput：包含下一帧 obs、reward、done、info

        Raises:
            ValueError: action 超出合法范围
            RuntimeError: 环境内部状态错误
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """释放环境资源"""
        ...

    def seed(self, seed: int) -> None:
        """
        设置环境随机种子。

        Args:
            seed: 随机种子

        Default: 不做处理（子类可覆盖）
        """
        pass
