"""
DA-OCL Dummy 测试环境
用于在没有真实环境时进行功能验证
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from da_ocl.core.constants import ACTION_DIM, OBS_DIM
from envs.base_env import BaseEnv, EnvOutput


class DummyEnvironment(BaseEnv):
    """
    Dummy 测试环境。

    生成随机观测，支持配置实体数量、观测维度、动作空间大小。
    模拟死亡事件（通过 r_penalty > DEATH_R_PENALTY_THRESHOLD 触发）。

    用于：
    - 单元测试
    - 快速功能验证
    - 集成测试
    """

    def __init__(
        self,
        num_entities: int = 4,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        max_episode_steps: int = 1000,
        seed: int | None = None,
        death_probability: float = 0.01,
    ) -> None:
        self._num_entities = num_entities
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_episode_steps = max_episode_steps
        self._death_probability = death_probability

        self._step_count: int = 0
        self._rng = np.random.RandomState(seed)

        self._current_obs: torch.Tensor | None = None

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def num_entities(self) -> int:
        return self._num_entities

    def reset(self) -> EnvOutput:
        """重置环境，返回初始观测"""
        self._step_count = 0
        obs = self._generate_obs()
        self._current_obs = obs

        return EnvOutput(
            obs=obs,
            reward=0.0,
            done=False,
            info={"episode_step": 0},
        )

    def step(self, action: int) -> EnvOutput:
        """执行一个动作，返回下一帧"""
        if not (0 <= action < self._action_dim):
            raise ValueError(f"action {action} 超出合法范围 [0, {self._action_dim})")

        self._step_count += 1

        # 模拟死亡事件（概率触发）
        is_death = self._rng.random() < self._death_probability

        obs = self._generate_obs()
        self._current_obs = obs

        if is_death:
            reward = -10.0
            done = True
            info = {
                "episode_step": self._step_count,
                "is_death": True,
                "r_penalty": 10.0,
                "death_distance": 0,
            }
        else:
            # 稀疏奖励
            reward = self._rng.randn() * 0.1  # 小幅波动奖励
            done = self._step_count >= self._max_episode_steps
            info = {
                "episode_step": self._step_count,
                "is_death": False,
                "r_penalty": 0.0,
            }

        return EnvOutput(
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def _generate_obs(self) -> torch.Tensor:
        """生成随机观测向量"""
        obs = torch.from_numpy(self._rng.randn(self._obs_dim)).float()
        return obs

    def close(self) -> None:
        """释放环境资源（Dummy 环境无需释放）"""
        pass

    def seed(self, seed: int) -> None:
        """设置随机种子"""
        self._rng = np.random.RandomState(seed)

    @property
    def step_count(self) -> int:
        """当前 episode 的步数"""
        return self._step_count
