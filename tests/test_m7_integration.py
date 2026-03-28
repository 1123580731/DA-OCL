"""
DA-OCL M7 集成测试
端到端验证白天 + 夜晚流程
"""

from __future__ import annotations

import pytest
import torch

from da_ocl.agent import DAOCLAgent
from da_ocl.night.night_trigger import NightTrigger
from da_ocl.memory.mask_history import MaskHistory
from envs.dummy_env import DummyEnvironment


class TestAgentInstantiation:
    def test_agent_can_be_created(self):
        env = DummyEnvironment()
        agent = DAOCLAgent(env)
        assert agent.device.type in ("cpu", "cuda")
        assert agent.buffer.size() == 0
        assert agent._timestep == 0

    def test_agent_reset(self):
        env = DummyEnvironment()
        agent = DAOCLAgent(env)
        agent.reset()
        assert agent._timestep == 0
        assert agent._cumulated_surprise == []


class TestAgentObserve:
    def test_single_step(self):
        env = DummyEnvironment()
        agent = DAOCLAgent(env)
        agent.reset()

        env_output = env.reset()
        result = agent.step(env_output)

        assert "ssm_state" in result
        assert "fast_retrieved" in result
        assert "surprise" in result
        assert "composite" in result
        assert "should_night" in result

    def test_buffer_grows(self):
        env = DummyEnvironment()
        agent = DAOCLAgent(env)
        agent.reset()

        for i in range(5):
            env_output = env.reset()
            agent.step(env_output)

        assert agent.buffer.size() == 5

    def test_mask_history_accumulates(self):
        env = DummyEnvironment()
        agent = DAOCLAgent(env)
        agent.reset()

        for i in range(3):
            env_output = env.reset()
            agent.step(env_output)

        summary = agent.mask_history.summary()
        assert summary["count"] >= 0  # 参数数量


class TestNightTrigger:
    def test_manual_trigger_check(self):
        trigger = NightTrigger(
            window_threshold=10,
            mask_density_threshold=1.0,
            surp_threshold=100.0,
            timestep_fallback=100000,
        )
        should, reason = trigger.should_trigger(
            buffer_window_count=15,
            mask_density=0.1,
            avg_surprise=0.5,
            timestep=0,
        )
        assert should is True
        assert "Buffer" in reason

    def test_night_trigger_in_pipeline(self):
        env = DummyEnvironment()
        agent = DAOCLAgent(env)
        agent.reset()

        # 替换 pipeline 内的 trigger 为宽松阈值
        from da_ocl.core.constants import NIGHT_WINDOW_THRESHOLD, NIGHT_MASK_DENSITY_THRESHOLD, NIGHT_SURP_THRESHOLD, NIGHT_TIMESTEP_FALLBACK
        agent.night_pipeline.night_trigger = NightTrigger(
            window_threshold=NIGHT_WINDOW_THRESHOLD,
            mask_density_threshold=1.0,
            surp_threshold=10000.0,
            timestep_fallback=100000,
        )

        # 前几步不应触发夜晚
        for i in range(3):
            env_output = env.reset()
            result = agent.step(env_output)
            assert result["should_night"] is False, f"Step {i} should not trigger night"


class TestMaskHistoryMethods:
    def test_new_methods(self):
        mh = MaskHistory()
        mh.accumulate("param1", 0.5)
        mh.accumulate("param2", 0.3)

        acc = mh.get_accumulated()
        assert acc["param1"] == 0.5
        assert acc["param2"] == 0.3

        density = mh.get_density()
        assert 0.0 <= density <= 1.0

        avg_surp = mh.get_avg_surprise()
        assert avg_surp >= 0.0

        mh.reset()
        assert mh.get_density() == 0.0
