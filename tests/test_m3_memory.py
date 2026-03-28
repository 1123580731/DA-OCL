"""
M3 测试：双记忆系统
测试快记忆、慢记忆、联合掩码、掩码历史
"""

from __future__ import annotations

import pytest
import torch

from da_ocl.core.constants import (
    COMPOSITE_DIM,
    ENTITY_DIM,
    GNN_HIDDEN_DIM,
    RULE_DIM,
    SSM_STATE_DIM,
    WINDOW_SIZE_DEFAULT,
)
from da_ocl.memory.fast_memory import FastMemory
from da_ocl.memory.slow_memory import SlowMemory
from da_ocl.memory.joint_mask import compute_joint_mask
from da_ocl.memory.mask_history import MaskHistory


class TestFastMemory:
    """测试快记忆"""

    def test_fast_memory_store_single(self, torch_device):
        """存储单个复合特征"""
        fm = FastMemory(capacity=100, composite_dim=COMPOSITE_DIM).to(torch_device)
        assert fm.stored_count == 0
        assert fm.is_empty

        composite = torch.randn(COMPOSITE_DIM, device=torch_device)
        fm.store(composite)
        assert fm.stored_count == 1
        assert not fm.is_empty

    def test_fast_memory_store_batch(self, torch_device):
        """存储多个复合特征（填充到容量）"""
        fm = FastMemory(capacity=5, composite_dim=COMPOSITE_DIM).to(torch_device)

        for i in range(5):
            composite = torch.randn(COMPOSITE_DIM, device=torch_device) * (i + 1)
            fm.store(composite)

        assert fm.stored_count == 5

    def test_fast_memory_retrieve(self, torch_device):
        """检索返回正确 shape"""
        fm = FastMemory(capacity=10, composite_dim=COMPOSITE_DIM).to(torch_device)

        # 存储一些特征
        originals = []
        for i in range(5):
            comp = torch.randn(COMPOSITE_DIM, device=torch_device)
            fm.store(comp)
            originals.append(comp)

        # 检索（查询与存储最相似的）
        query = originals[2].clone()
        best_match, attn = fm.retrieve(query)
        assert best_match.shape == (COMPOSITE_DIM,)
        assert attn.shape[0] == 5  # 5 个存储槽

    def test_fast_memory_retrieve_empty_raises(self, torch_device):
        """空记忆检索应抛出 RuntimeError"""
        fm = FastMemory(capacity=10, composite_dim=COMPOSITE_DIM).to(torch_device)
        with pytest.raises(RuntimeError, match="空"):
            fm.retrieve(torch.randn(COMPOSITE_DIM, device=torch_device))

    def test_fast_memory_capacity_overflow(self, torch_device):
        """容量溢出时 FIFO 替换"""
        fm = FastMemory(capacity=3, composite_dim=COMPOSITE_DIM).to(torch_device)

        for i in range(5):
            fm.store(torch.randn(COMPOSITE_DIM, device=torch_device) * (i + 1))

        assert fm.stored_count == 3  # 容量固定为 3

    def test_fast_memory_get_distribution(self, torch_device):
        """软检索分布"""
        fm = FastMemory(capacity=10, composite_dim=COMPOSITE_DIM).to(torch_device)
        for _ in range(5):
            fm.store(torch.randn(COMPOSITE_DIM, device=torch_device))

        query = torch.randn(COMPOSITE_DIM, device=torch_device)
        best_match, probs = fm.get_retrieval_distribution(query)
        assert best_match.shape == (COMPOSITE_DIM,)
        assert probs.shape[0] == 5
        assert abs(probs.sum().item() - 1.0) < 1e-5  # 概率和为 1

    def test_fast_memory_state_dict(self, torch_device):
        """存储/恢复状态"""
        fm = FastMemory(capacity=10, composite_dim=COMPOSITE_DIM).to(torch_device)
        for _ in range(3):
            fm.store(torch.randn(COMPOSITE_DIM, device=torch_device))

        state = fm.get_state_dict()
        assert state["keys"].shape[0] == 3
        assert state["_stored_count_local"] == 3

        # 创建新实例并加载
        fm2 = FastMemory(capacity=10, composite_dim=COMPOSITE_DIM).to(torch_device)
        fm2.load_state_dict(state)
        assert fm2.stored_count == 3


class TestSlowMemory:
    """测试慢记忆"""

    def test_slow_memory_forward_shape(self, torch_device):
        """三层串联：输入 (T, entity_dim) → 输出 (T, h, r)"""
        sm = SlowMemory(use_mamba=False).to(torch_device)
        T = WINDOW_SIZE_DEFAULT
        entity_window = torch.randn(T, ENTITY_DIM, device=torch_device)

        gnn_out, ssm_out, rule_out = sm(entity_window)

        assert gnn_out.shape == (T, GNN_HIDDEN_DIM)
        assert ssm_out.shape == (SSM_STATE_DIM,)
        assert rule_out.shape == (RULE_DIM,)

    def test_slow_memory_retrieve(self, torch_device):
        """检索返回 SSM 表征"""
        sm = SlowMemory(use_mamba=False).to(torch_device)
        entity_window = torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device)

        o_hat = sm.retrieve(entity_window)
        assert o_hat.shape == (SSM_STATE_DIM,)

    def test_slow_memory_silent_write(self, torch_device):
        """沉默写入不报错（无梯度更新）"""
        sm = SlowMemory(use_mamba=False).to(torch_device)
        entity_window = torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device)

        # 掩码为 0：不记录
        sm.silent_write(entity_window, mask_value=0.0)
        assert len(sm.get_mask_intensity()) == 0

        # 掩码 > 0：记录掩码强度
        sm.silent_write(entity_window, mask_value=0.5)
        intensities = sm.get_mask_intensity()
        assert len(intensities) > 0
        assert all(0 < v <= 1.0 for v in intensities.values())

    def test_slow_memory_mask_max_accumulation(self, torch_device):
        """掩码强度取逐参数 max"""
        sm = SlowMemory(use_mamba=False).to(torch_device)
        entity_window = torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device)

        sm.silent_write(entity_window, mask_value=0.3)
        sm.silent_write(entity_window, mask_value=0.7)  # 更高
        sm.silent_write(entity_window, mask_value=0.5)  # 更低（不覆盖）

        intensities = sm.get_mask_intensity()
        assert all(v == 0.7 for v in intensities.values())

    def test_slow_memory_clear_intensity(self, torch_device):
        """清空掩码强度"""
        sm = SlowMemory(use_mamba=False).to(torch_device)
        entity_window = torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device)
        sm.silent_write(entity_window, mask_value=0.5)
        assert len(sm.get_mask_intensity()) > 0

        sm.clear_mask_intensity()
        assert len(sm.get_mask_intensity()) == 0


class TestJointMask:
    """测试联合掩码计算"""

    def test_mask_in_bounds(self, torch_device):
        """掩码值 ∈ [0, 1]"""
        surprise = torch.tensor([0.5, 1.0, 2.0], device=torch_device)
        mask = compute_joint_mask(surprise, coverage_rate=0.3)
        assert 0.0 <= mask <= 1.0

    def test_mask_zero_coverage(self, torch_device):
        """完全无规律覆盖时掩码最大（接近 0.5，上限由 ReLU 决定）"""
        surprise = torch.tensor([10.0], device=torch_device)
        mask = compute_joint_mask(surprise, coverage_rate=0.0)
        # max(sigmoid(10)-0.5, 0) * 1.0 ≈ 0.5
        assert mask > 0.4  # 约 0.5

    def test_mask_full_coverage(self, torch_device):
        """完全覆盖时掩码为零"""
        surprise = torch.tensor([10.0], device=torch_device)
        mask = compute_joint_mask(surprise, coverage_rate=1.0)
        assert mask == pytest.approx(0.0, abs=1e-3)

    def test_mask_zero_surprise(self, torch_device):
        """零惊喜度时掩码为零（无认知冲突）"""
        surprise = torch.tensor([0.0], device=torch_device)
        mask = compute_joint_mask(surprise, coverage_rate=0.5)
        assert mask == pytest.approx(0.0, abs=1e-3)

    def test_mask_uses_max_not_mean(self, torch_device):
        """掩码使用 max 而非 mean"""
        # 一个峰值，其余为零
        surprise = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0], device=torch_device)
        mask_max = compute_joint_mask(surprise, coverage_rate=0.0)

        # 全相同
        surprise2 = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], device=torch_device)
        mask_mean = compute_joint_mask(surprise2, coverage_rate=0.0)

        # 峰值掩码应大于均匀掩码
        assert mask_max > mask_mean

    def test_mask_coverage_rate_out_of_range_raises(self, torch_device):
        """coverage_rate 超出 [0,1] 抛出 ValueError"""
        surprise = torch.tensor([1.0], device=torch_device)
        with pytest.raises(ValueError):
            compute_joint_mask(surprise, coverage_rate=-0.1)
        with pytest.raises(ValueError):
            compute_joint_mask(surprise, coverage_rate=1.5)


class TestMaskHistory:
    """测试掩码历史累积器"""

    def test_accumulate_basic(self):
        """累积基本行为"""
        mh = MaskHistory()
        mh.accumulate("param_a", 0.5)
        mh.accumulate("param_b", 0.3)

        assert mh.get_mask("param_a") == 0.5
        assert mh.get_mask("param_b") == 0.3
        assert mh.total_activated == 2

    def test_accumulate_max_behavior(self):
        """累积取 max"""
        mh = MaskHistory()
        mh.accumulate("param", 0.3)
        mh.accumulate("param", 0.5)  # 更高
        mh.accumulate("param", 0.2)  # 更低（不覆盖）

        assert mh.get_mask("param") == 0.5

    def test_get_activated_params(self):
        """获取激活参数"""
        mh = MaskHistory()
        mh.accumulate("param_a", 0.5)
        mh.accumulate("param_b", 0.01)  # 极低
        mh.accumulate("param_c", 0.3)

        activated = mh.get_activated_params(threshold=0.1)
        assert "param_a" in activated
        assert "param_b" not in activated  # 低于阈值
        assert "param_c" in activated

    def test_merge(self):
        """合并两个掩码历史"""
        mh1 = MaskHistory()
        mh1.accumulate("param_a", 0.5)
        mh1.accumulate("param_b", 0.3)

        mh2 = MaskHistory()
        mh2.accumulate("param_a", 0.2)  # 更低
        mh2.accumulate("param_c", 0.7)

        mh1.merge(mh2)
        assert mh1.get_mask("param_a") == 0.5  # 保留更大的
        assert mh1.get_mask("param_b") == 0.3
        assert mh1.get_mask("param_c") == 0.7

    def test_clear(self):
        """清空历史"""
        mh = MaskHistory()
        mh.accumulate("param", 0.5)
        mh.clear()
        assert mh.total_activated == 0
        assert mh.get_mask("param") == 0.0

    def test_summary(self):
        """摘要统计"""
        mh = MaskHistory()
        summary = mh.summary()
        assert summary["count"] == 0
        assert summary["max"] == 0.0

        mh.accumulate("a", 0.1)
        mh.accumulate("b", 0.5)
        summary = mh.summary()
        assert summary["count"] == 2
        assert summary["max"] == 0.5
        assert summary["mean"] == pytest.approx(0.3)

    def test_accumulate_from_model(self, torch_device):
        """从模型批量累积"""
        sm = SlowMemory(use_mamba=False).to(torch_device)
        mh = MaskHistory.accumulate_from_model(sm, mask_value=0.4)
        assert mh.total_activated > 0
        assert all(v == 0.4 for v in mh._history.values())
