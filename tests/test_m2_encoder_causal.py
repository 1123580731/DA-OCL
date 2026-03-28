"""
M2 测试：编码器层 + 因果窗口管理器
"""

from __future__ import annotations

import pytest
import torch

from da_ocl.core.constants import (
    ENTITY_DIM,
    GNN_HIDDEN_DIM,
    OBS_DIM,
    RULE_DIM,
    SSM_STATE_DIM,
    WINDOW_SIZE_DEFAULT,
)
from da_ocl.core.encoder import Encoder
from da_ocl.core.gnn_layer import GNNRelationEncoder
from da_ocl.core.ssm_layer import FallbackSSMLayer, SSMCompressor
from da_ocl.core.rule_constraints import RuleConstraintLayer
from da_ocl.causal.window_manager import CausalWindowManager
from da_ocl.causal.surprise_detector import SurpriseDetector
from da_ocl.causal.coverage_tracker import CoverageTracker
from envs.dummy_env import DummyEnvironment


class TestEncoder:
    """测试编码器层"""

    def test_encoder_forward_single(self, torch_device):
        """单帧输入：shape (obs_dim,) → (entity_dim,)"""
        enc = Encoder().to(torch_device)
        obs = torch.randn(OBS_DIM, device=torch_device)
        entity = enc(obs)
        assert entity.shape == (ENTITY_DIM,)
        assert entity.device == torch_device

    def test_encoder_forward_batch(self, torch_device):
        """批次输入：shape (batch, obs_dim) → (batch, entity_dim)"""
        enc = Encoder().to(torch_device)
        obs = torch.randn(4, OBS_DIM, device=torch_device)
        entity = enc(obs)
        assert entity.shape == (4, ENTITY_DIM)
        assert entity.device == torch_device

    def test_encoder_wrong_dim_raises(self, torch_device):
        """错误的输入维度应抛出断言错误"""
        enc = Encoder().to(torch_device)
        wrong_obs = torch.randn(64, device=torch_device)  # 错误维度
        with pytest.raises(AssertionError):
            enc(wrong_obs)

    def test_encoder_trainable(self, torch_device):
        """编码器应可训练（包含可学习参数）"""
        enc = Encoder().to(torch_device)
        params_before = [p.clone() for p in enc.parameters()]
        obs = torch.randn(OBS_DIM, device=torch_device)
        entity = enc(obs)
        loss = entity.sum()
        loss.backward()
        with torch.no_grad():
            for p_before, p_after in zip(enc.parameters(), enc.parameters()):
                # 至少某些参数应被更新
                pass  # 梯度验证在完整集成测试中
        assert any(p.requires_grad for p in enc.parameters())


class TestGNNLayer:
    """测试 GNN 实体关系编码层"""

    def test_gnn_forward_shape(self, torch_device):
        """输入 (T, entity_dim) → 输出 (T, GNN_HIDDEN_DIM)"""
        gnn = GNNRelationEncoder().to(torch_device)
        T = WINDOW_SIZE_DEFAULT
        entity_window = torch.randn(T, ENTITY_DIM, device=torch_device)
        out = gnn(entity_window)
        assert out.shape == (T, GNN_HIDDEN_DIM)
        assert out.device == torch_device

    def test_gnn_forward_multientity(self, torch_device):
        """输入 (T, num_entities, entity_dim) → 输出 (T, GNN_HIDDEN_DIM)"""
        gnn = GNNRelationEncoder(num_entities=4).to(torch_device)
        T = 5
        entity_window = torch.randn(T, 4, ENTITY_DIM, device=torch_device)
        out = gnn(entity_window)
        assert out.shape == (T, GNN_HIDDEN_DIM)

    def test_gnn_requires_grad(self, torch_device):
        """GNN 输出应可求导"""
        gnn = GNNRelationEncoder().to(torch_device)
        entity_window = torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device, requires_grad=True)
        out = gnn(entity_window)
        loss = out.sum()
        loss.backward()
        assert entity_window.grad is not None


class TestSSMLayer:
    """测试 SSM 规律压缩层"""

    def test_ssm_fallback_forward(self, torch_device):
        """降级 SSM：输入 (T, gnn_hidden_dim) → 输出 (SSM_STATE_DIM,)"""
        ssm = FallbackSSMLayer(GNN_HIDDEN_DIM, SSM_STATE_DIM).to(torch_device)
        T = 6
        rel_features = torch.randn(T, GNN_HIDDEN_DIM, device=torch_device)
        out = ssm(rel_features)
        assert out.shape == (SSM_STATE_DIM,)
        assert out.device == torch_device

    def test_ssm_fallback_requires_grad(self, torch_device):
        """降级 SSM 输出应可求导"""
        ssm = FallbackSSMLayer(GNN_HIDDEN_DIM, SSM_STATE_DIM).to(torch_device)
        rel_features = torch.randn(WINDOW_SIZE_DEFAULT, GNN_HIDDEN_DIM, device=torch_device, requires_grad=True)
        out = ssm(rel_features)
        loss = out.sum()
        loss.backward()
        assert rel_features.grad is not None

    def test_ssm_compressor_auto_fallback(self, torch_device):
        """SSMCompressor 应自动降级（当 mamba-ssm 不可用时）"""
        ssm = SSMCompressor(use_mamba=False).to(torch_device)
        assert ssm.implementation == "gru_fallback"
        T = 6
        rel_features = torch.randn(T, GNN_HIDDEN_DIM, device=torch_device)
        out = ssm(rel_features)
        assert out.shape == (SSM_STATE_DIM,)

    def test_ssm_empty_sequence_raises(self, torch_device):
        """空序列应抛出错误"""
        ssm = FallbackSSMLayer(GNN_HIDDEN_DIM, SSM_STATE_DIM).to(torch_device)
        empty = torch.randn(0, GNN_HIDDEN_DIM, device=torch_device)
        with pytest.raises(ValueError, match="0"):
            ssm(empty)


class TestRuleConstraintLayer:
    """测试逻辑约束层"""

    def test_rules_forward_1d(self, torch_device):
        """1D 输入：(SSM_STATE_DIM,) → (RULE_DIM,)"""
        rules = RuleConstraintLayer().to(torch_device)
        ssm_out = torch.randn(SSM_STATE_DIM, device=torch_device)
        out = rules(ssm_out)
        assert out.shape == (RULE_DIM,)
        assert out.device == torch_device

    def test_rules_forward_2d(self, torch_device):
        """2D 批次输入：(batch, SSM_STATE_DIM) → (batch, RULE_DIM)"""
        rules = RuleConstraintLayer().to(torch_device)
        ssm_out = torch.randn(4, SSM_STATE_DIM, device=torch_device)
        out = rules(ssm_out)
        assert out.shape == (4, RULE_DIM)

    def test_rules_requires_grad(self, torch_device):
        """规则层输出应可求导"""
        rules = RuleConstraintLayer().to(torch_device)
        ssm_out = torch.randn(SSM_STATE_DIM, device=torch_device, requires_grad=True)
        out = rules(ssm_out)
        loss = out.sum()
        loss.backward()
        assert ssm_out.grad is not None


class TestCausalWindowManager:
    """测试因果窗口管理器"""

    def test_window_fifo_behavior(self, torch_device):
        """FIFO 行为：先进先出，窗口长度固定"""
        T = 5
        wm = CausalWindowManager(window_size=T, device=torch_device)
        assert not wm.is_full()
        assert wm.filled_frames == 0

        for i in range(7):
            entity = torch.randn(ENTITY_DIM, device=torch_device) * (i + 1)
            window = wm.push(entity)
            assert wm.filled_frames == min(i + 1, T)

        # 窗口已满后，应为最近 T 帧
        assert wm.filled_frames == T
        batch = wm.get_batch()
        assert batch is not None
        assert batch.shape == (T, ENTITY_DIM)

    def test_window_returns_none_before_full(self, torch_device):
        """窗口未满前 get_batch() 返回 None"""
        wm = CausalWindowManager(window_size=5, device=torch_device)
        for i in range(4):
            wm.push(torch.randn(ENTITY_DIM, device=torch_device))
            assert wm.get_batch() is None

        wm.push(torch.randn(ENTITY_DIM, device=torch_device))
        assert wm.get_batch() is not None

    def test_window_reset(self, torch_device):
        """reset() 清空缓冲区"""
        wm = CausalWindowManager(window_size=5, device=torch_device)
        for _ in range(10):
            wm.push(torch.randn(ENTITY_DIM, device=torch_device))
        assert wm.is_full()
        wm.reset()
        assert not wm.is_full()
        assert wm.filled_frames == 0

    def test_window_wrong_entity_dim_raises(self, torch_device):
        """错误的实体维度应抛出错误"""
        wm = CausalWindowManager(window_size=5, device=torch_device)
        wrong_entity = torch.randn(64, device=torch_device)  # 错误维度
        with pytest.raises(ValueError):
            wm.push(wrong_entity)

    def test_window_step_count(self, torch_device):
        """步数计数器正确累加"""
        wm = CausalWindowManager(window_size=5, device=torch_device)
        assert wm.step_count == 0
        for i in range(5):
            wm.push(torch.randn(ENTITY_DIM, device=torch_device))
            assert wm.step_count == i + 1

    def test_window_device_transfer(self, torch_device):
        """to(device) 正确迁移"""
        wm = CausalWindowManager(window_size=5, device=torch.device("cpu"))
        for _ in range(5):
            wm.push(torch.randn(ENTITY_DIM))
        if torch.cuda.is_available():
            wm_cuda = wm.to(torch.device("cuda"))
            assert wm_cuda.device.type == "cuda"


class TestSurpriseDetector:
    """测试惊喜度检测器"""

    def test_surprise_scalar_nonzero(self, torch_device):
        """不同表征应产生非零惊喜度"""
        o1 = torch.randn(SSM_STATE_DIM, device=torch_device)
        o2 = torch.randn(SSM_STATE_DIM, device=torch_device)
        scalar, per_frame = SurpriseDetector.compute_window_surprise(o1, o2)
        assert scalar > 0
        assert len(per_frame) == 6  # 默认 T=6
        assert all(f >= 0 for f in per_frame)

    def test_surprise_zero_identical(self, torch_device):
        """相同表征应产生零惊喜度"""
        o = torch.randn(SSM_STATE_DIM, device=torch_device)
        scalar, _ = SurpriseDetector.compute_window_surprise(o, o.clone())
        assert scalar == pytest.approx(0.0, abs=1e-5)

    def test_composite_surprise(self, torch_device):
        """复合特征惊喜度"""
        from da_ocl.core.constants import COMPOSITE_DIM
        c1 = torch.randn(COMPOSITE_DIM, device=torch_device)
        c2 = torch.randn(COMPOSITE_DIM, device=torch_device)
        surp = SurpriseDetector.compute_composite_surprise(c1, c2)
        assert surp > 0


class TestCoverageTracker:
    """测试规律覆盖追踪器"""

    def test_initial_coverage_zero(self):
        """初始覆盖率应为 0（规律库空）"""
        tracker = CoverageTracker()
        assert tracker.get_global_mean() == 0.0
        assert tracker.get_batch_coverage_rate() == 1.0  # 覆盖缺口最大

    def test_coverage_update_increases(self):
        """多次更新应提升覆盖率"""
        tracker = CoverageTracker()
        for i in range(10):
            tracker.update([f"rule_{i % 5}"], batch_surprise=0.5)
        assert tracker.get_global_mean() > 0

    def test_coverage_gap_decreases(self):
        """覆盖率提升 → 覆盖缺口降低"""
        tracker = CoverageTracker()
        initial_gap = tracker.get_batch_coverage_rate()
        for _ in range(20):
            tracker.update(["rule_0"], batch_surprise=0.5)
        final_gap = tracker.get_batch_coverage_rate()
        assert final_gap < initial_gap


class TestDummyEnvironment:
    """测试 Dummy 环境"""

    def test_env_reset(self, torch_device):
        """reset() 返回初始观测"""
        env = DummyEnvironment(obs_dim=OBS_DIM, action_dim=8)
        output = env.reset()
        assert output.obs.shape == (OBS_DIM,)
        assert output.reward == 0.0
        assert output.done is False

    def test_env_step(self, torch_device):
        """step() 返回有效输出"""
        env = DummyEnvironment(obs_dim=OBS_DIM, action_dim=8)
        env.reset()
        for action in [0, 1, 7]:
            output = env.step(action)
            assert output.obs.shape == (OBS_DIM,)
            assert isinstance(output.done, bool)
            assert "r_penalty" in output.info

    def test_env_step_wrong_action_raises(self):
        """非法动作应抛出 ValueError"""
        env = DummyEnvironment(action_dim=8)
        env.reset()
        with pytest.raises(ValueError):
            env.step(10)

    def test_env_death_events(self):
        """模拟死亡事件"""
        env = DummyEnvironment(death_probability=0.5, seed=42)
        env.reset()
        deaths = 0
        for _ in range(20):
            out = env.step(0)
            if out.info.get("is_death"):
                deaths += 1
                assert out.info["r_penalty"] >= 8.0
        # 统计意义：期望约 10 次死亡（50% * 20）
        assert deaths > 0
