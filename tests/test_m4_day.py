"""
M4 测试：白天流程
测试复合特征构建器、对抗滤波、双指标、情景边界检测
"""

from __future__ import annotations

import pytest
import torch

from da_ocl.core.composite_builder import CompositeFeatureBuilder
from da_ocl.core.constants import COMPOSITE_DIM, SSM_STATE_DIM, WINDOW_SIZE_DEFAULT
from da_ocl.day.adversarial_filter import AdversarialFilter, FilterResult
from da_ocl.day.metrics import (
    compute_importance,
    compute_surprise,
    is_death_experience,
    compute_task_value,
)
from da_ocl.causal.boundary_detector import BoundaryDetector
from da_ocl.causal.surprise_detector import SurpriseDetector


class TestCompositeBuilder:
    """测试复合特征构建器"""

    def test_composite_shape(self, torch_device):
        """复合特征 shape = (3*SSM_STATE_DIM,)"""
        o = torch.randn(SSM_STATE_DIM, device=torch_device)
        o_hat = torch.randn(SSM_STATE_DIM, device=torch_device)
        composite = CompositeFeatureBuilder.build(o, o_hat)
        assert composite.shape == (COMPOSITE_DIM,)
        assert composite.device == torch_device

    def test_composite_decompose_roundtrip(self, torch_device):
        """分解后重新拼接应还原原始值"""
        o = torch.randn(SSM_STATE_DIM, device=torch_device)
        o_hat = torch.randn(SSM_STATE_DIM, device=torch_device)
        composite = CompositeFeatureBuilder.build(o, o_hat)
        o2, o_hat2, residual2 = CompositeFeatureBuilder.decompose(composite)
        assert torch.allclose(o, o2, atol=1e-6)
        assert torch.allclose(o_hat, o_hat2, atol=1e-6)
        assert torch.allclose(o - o_hat, residual2, atol=1e-6)

    def test_composite_residual_semantics(self, torch_device):
        """残差 = O - O_hat"""
        o = torch.randn(SSM_STATE_DIM, device=torch_device)
        o_hat = torch.randn(SSM_STATE_DIM, device=torch_device)
        composite = CompositeFeatureBuilder.build(o, o_hat)
        o2, o_hat2, residual = CompositeFeatureBuilder.decompose(composite)
        assert torch.allclose(residual, o - o_hat, atol=1e-5)


class TestAdversarialFilter:
    """测试对抗滤波"""

    def test_filter_passes_reasonable_input(self, torch_device):
        """合理的惊喜度应通过滤波"""
        af = AdversarialFilter()
        composite = torch.randn(COMPOSITE_DIM, device=torch_device)
        surprise = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2], device=torch_device)  # 都很低
        result = af.filter(composite, surprise)
        assert isinstance(result, FilterResult)
        assert result.coarse_passed
        assert result.passed

    def test_filter_rejects_high_surprise(self, torch_device):
        """极高惊喜度应被拒绝"""
        af = AdversarialFilter()
        composite = torch.randn(COMPOSITE_DIM, device=torch_device)
        surprise = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0], device=torch_device)
        result = af.filter(composite, surprise)
        assert not result.passed


class TestMetrics:
    """测试双指标计算"""

    def test_importance_formula(self, torch_device):
        """Imp = alpha*(1-coverage) + (1-alpha)*r_penalty"""
        imp = compute_importance(coverage_rate=0.5, r_penalty=0.0)
        assert 0.0 <= imp <= 1.0
        assert imp == pytest.approx(0.5 * 0.7)  # 0.7 * 0.5 + 0.3 * 0 = 0.35

    def test_importance_death_threshold(self, torch_device):
        """r_penalty 超过阈值应触发保护区逻辑"""
        assert is_death_experience(r_penalty=8.0)
        assert is_death_experience(r_penalty=10.0)
        assert not is_death_experience(r_penalty=5.0)
        assert not is_death_experience(r_penalty=0.0)

    def test_surprise_scalar_nonzero(self, torch_device):
        """不同表征产生非零惊喜度"""
        o1 = torch.randn(SSM_STATE_DIM, device=torch_device)
        o2 = torch.randn(SSM_STATE_DIM, device=torch_device)
        scalar, per_frame = compute_surprise(o1, o2)
        assert scalar > 0
        assert per_frame.shape[0] == WINDOW_SIZE_DEFAULT

    def test_task_value(self, torch_device):
        """任务价值分数应返回有限值"""
        composite = torch.randn(COMPOSITE_DIM, device=torch_device)
        rule_out = torch.randn(256, device=torch_device)
        task_val = compute_task_value(composite, rule_out, reward=1.0)
        assert isinstance(task_val, float)
        assert 0.0 <= task_val <= 100.0  # 宽松范围


class TestBoundaryDetector:
    """测试情景边界检测器"""

    def test_no_boundary_during_warmup(self, torch_device):
        """预热期（前 10 个 batch）不触发边界"""
        bd = BoundaryDetector()
        for i in range(9):
            assert not bd.detect(float(i))
        assert bd.boundary_count == 0

    def test_boundary_triggers_on_surprise_jump(self, torch_device):
        """惊喜度突变应触发边界"""
        bd = BoundaryDetector()
        # 先填满预热期
        for _ in range(10):
            bd.detect(1.0)
        # 突然大幅跳升
        triggered = bd.detect(10.0)
        assert triggered
        assert bd.boundary_count == 1
        assert bd.is_new_boundary()

    def test_reset_clears_state(self, torch_device):
        """reset() 清空状态"""
        bd = BoundaryDetector(history_size=20)
        # 填满暖身期（前10个batch不触发边界）
        for _ in range(10):
            bd.detect(1.0)
        # 触发边界
        for _ in range(5):
            bd.detect(10.0)
        assert bd.boundary_count > 0
        bd.reset()
        assert bd.boundary_count == 0
        assert bd.history_mean == 0.0

    def test_history_stats(self, torch_device):
        """历史统计正确"""
        bd = BoundaryDetector()
        for i in range(15):
            bd.detect(float(i))
        assert bd.history_mean > 0
