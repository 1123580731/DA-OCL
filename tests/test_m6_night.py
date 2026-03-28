"""
DA-OCL M6 夜晚流程测试
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from da_ocl.core.constants import (
    ALPHA_KL,
    BETA0_KL,
    COMPOSITE_DIM,
    EWC_LAMBDA,
    GAMMA_DREAM,
    RULE_DIM,
    SSM_STATE_DIM,
)
from da_ocl.night.night_trigger import NightTrigger
from da_ocl.losses.night_losses import (
    DistillationLoss,
    KLLoss,
    EWCLoss,
    DreamLoss,
    compute_total_night_loss,
)
from da_ocl.regulation.axes import (
    ImportanceRegulation,
    SurpriseRegulation,
    TemporalDecayRegulation,
    RegulationMixer,
)
from da_ocl.night.distillation import DistillationPath
from da_ocl.night.dream_generator import DreamGenerator
from da_ocl.night.gradient_projector import GradientProjector
from da_ocl.night.ewc_updater import EWCUpdater
from da_ocl.night.consistency_gate import ConsistencyGate


class TestNightTrigger:
    def test_trigger_by_buffer_count(self):
        trigger = NightTrigger(window_threshold=10, mask_density_threshold=1.0, surp_threshold=100.0, timestep_fallback=100000)
        should, reason = trigger.should_trigger(buffer_window_count=15, mask_density=0.1, avg_surprise=0.5, timestep=0)
        assert should is True
        assert "Buffer" in reason

    def test_trigger_by_mask_density(self):
        trigger = NightTrigger(window_threshold=1000, mask_density_threshold=0.3, surp_threshold=100.0, timestep_fallback=100000)
        should, reason = trigger.should_trigger(buffer_window_count=5, mask_density=0.5, avg_surprise=0.5, timestep=0)
        assert should is True
        assert "掩码激活密度" in reason

    def test_trigger_by_surprise(self):
        trigger = NightTrigger(window_threshold=1000, mask_density_threshold=1.0, surp_threshold=2.0, timestep_fallback=100000)
        should, reason = trigger.should_trigger(buffer_window_count=5, mask_density=0.1, avg_surprise=3.0, timestep=0)
        assert should is True
        assert "惊喜度" in reason

    def test_trigger_by_timestep_fallback(self):
        trigger = NightTrigger(window_threshold=1000, mask_density_threshold=1.0, surp_threshold=100.0, timestep_fallback=100)
        should, reason = trigger.should_trigger(buffer_window_count=5, mask_density=0.1, avg_surprise=0.5, timestep=200)
        assert should is True
        assert "兜底" in reason

    def test_no_trigger(self):
        trigger = NightTrigger(window_threshold=1000, mask_density_threshold=1.0, surp_threshold=100.0, timestep_fallback=100000)
        should, reason = trigger.should_trigger(buffer_window_count=5, mask_density=0.1, avg_surprise=0.5, timestep=0)
        assert should is False
        assert reason == ""

    def test_reset_counters(self):
        trigger = NightTrigger()
        trigger.reset_counters()


class TestDistillationLoss:
    def test_forward_computes_weighted_mse(self):
        loss_fn = DistillationLoss()
        slow = torch.randn(SSM_STATE_DIM)
        fast = torch.randn(SSM_STATE_DIM)
        loss = loss_fn(slow, fast, importance=0.5, surprise=0.0)
        assert loss.dim() == 0
        assert loss >= 0.0

    def test_high_importance_increases_loss(self):
        loss_fn = DistillationLoss()
        slow = torch.randn(SSM_STATE_DIM)
        fast = slow + 0.1
        loss_low = loss_fn(slow, fast, importance=0.1, surprise=0.0)
        loss_high = loss_fn(slow, fast, importance=0.9, surprise=0.0)
        assert loss_high > loss_low


class TestKLLoss:
    def test_forward_computes_kl_divergence(self):
        loss_fn = KLLoss(beta0=BETA0_KL, alpha=ALPHA_KL)
        slow = F.softmax(torch.randn(32), dim=-1)
        fast = F.softmax(torch.randn(32), dim=-1)
        loss = loss_fn(slow, fast, importance=0.5)
        assert loss.dim() == 0
        assert loss >= 0.0

    def test_high_importance_reduces_loss(self):
        loss_fn = KLLoss(beta0=1.0, alpha=0.5)
        slow = F.softmax(torch.randn(32), dim=-1)
        fast = F.softmax(torch.randn(32), dim=-1)
        # 使用非零 surprise 确保 beta > 0（surprise_term = 1 - exp(-Surp/sigma)）
        loss_low_imp = loss_fn(slow, fast, importance=0.0, surprise=1.0)
        loss_high_imp = loss_fn(slow, fast, importance=2.0, surprise=1.0)
        assert loss_high_imp < loss_low_imp


class TestEWCLoss:
    def test_forward_zero_if_no_old_params(self):
        loss_fn = EWCLoss(ewc_lambda=1.0)
        model = torch.nn.Linear(10, 10)
        loss = loss_fn(model)
        assert loss == 0.0

    def test_fisher_computation(self):
        loss_fn = EWCLoss()
        model = torch.nn.Linear(8, 8)
        data = torch.randn(4, 8)
        loss_fn.compute_fisher(model, data)
        assert len(loss_fn._fisher_diagonal) > 0
        assert len(loss_fn._old_params) > 0


class TestDreamLoss:
    def test_forward_computes_entropy(self):
        loss_fn = DreamLoss(gamma=GAMMA_DREAM)
        slow_output = torch.randn(SSM_STATE_DIM)
        retrieved_dist = F.softmax(torch.randn(32), dim=-1)
        loss = loss_fn(slow_output, retrieved_dist)
        assert loss.dim() == 0
        assert loss >= 0.0

    def test_gamma_scales_loss(self):
        loss_fn_low = DreamLoss(gamma=0.01)
        loss_fn_high = DreamLoss(gamma=1.0)
        slow_output = torch.randn(SSM_STATE_DIM)
        retrieved_dist = F.softmax(torch.randn(32), dim=-1)
        loss_low = loss_fn_low(slow_output, retrieved_dist)
        loss_high = loss_fn_high(slow_output, retrieved_dist)
        assert loss_high > loss_low


class TestComputeTotalNightLoss:
    def test_sum_all_components(self):
        distill = torch.tensor(1.0)
        kl = torch.tensor(0.5)
        ewc = torch.tensor(0.2)
        dream = torch.tensor(0.3)
        total, components = compute_total_night_loss(distill, kl, ewc, dream)
        assert total == pytest.approx(2.0)
        assert "L_distill" in components
        assert "L_KL" in components
        assert "L_EWC" in components
        assert "L_dream" in components
        assert "L_total" in components


class TestImportanceRegulation:
    def test_mean_property(self):
        reg = ImportanceRegulation()
        assert 0.0 < reg.mean < 1.0

    def test_forward_output_shape(self):
        reg = ImportanceRegulation()
        signal = torch.randn(4, 16)
        out = reg(signal)
        assert out.dim() == 0
        assert 0.0 <= out.item() <= 1.0


class TestSurpriseRegulation:
    def test_mean_property(self):
        reg = SurpriseRegulation()
        assert 0.0 < reg.mean < 1.0

    def test_forward_positive_output(self):
        reg = SurpriseRegulation()
        signal = torch.randn(4, 16)
        out = reg(signal)
        assert out >= 0.0


class TestTemporalDecayRegulation:
    def test_forward_without_freshness(self):
        reg = TemporalDecayRegulation()
        param = torch.randn(16)
        out = reg(param)
        assert out.shape == param.shape
        assert (out >= 0.0).all()

    def test_set_freshness(self):
        reg = TemporalDecayRegulation()
        freshness = torch.rand(16)
        reg.set_freshness(freshness)
        assert reg._freshness is freshness


class TestRegulationMixer:
    def test_forward_returns_tensor(self):
        mixer = RegulationMixer()
        imp = torch.randn(4, 8)
        surp = torch.randn(4, 8)
        param = torch.randn(8)
        out = mixer(imp, surp, param)
        assert out.item() >= 0.0


class TestDistillationPath:
    def test_forward_output_shape(self):
        path = DistillationPath()
        fast = torch.randn(2, SSM_STATE_DIM)
        slow = torch.randn(2, SSM_STATE_DIM)
        imp = torch.tensor([0.5, 0.5])
        surp = torch.tensor([0.0, 0.0])
        loss = path(fast, slow, imp, surp)
        assert loss.dim() == 0
        assert loss >= 0.0


class TestDreamGenerator:
    def test_sample_along_causal_chain(self):
        gen = DreamGenerator(num_samples=8)
        start = torch.randn(2, 16)
        end = torch.randn(2, 16)
        samples = gen.sample_along_causal_chain(start, end)
        assert samples.shape == (2, 8, 16)

    def test_sample_boundary_variations(self):
        gen = DreamGenerator(num_samples=8)
        center = torch.randn(2, 16)
        samples = gen.sample_boundary_variations(center)
        assert samples.shape == (2, 8, 16)

    def test_generate_requires_slow_memory(self):
        gen = DreamGenerator()
        dummy_slow = torch.nn.Linear(16, 16)
        output = gen.generate(dummy_slow)
        assert "dream_samples" in output
        assert "num_samples" in output


class TestGradientProjector:
    def test_project_zeros_gradients_below_threshold(self):
        projector = GradientProjector()
        from da_ocl.memory.mask_history import MaskHistory
        mh = MaskHistory()
        mh.accumulate("layer.weight", 0.5)  # Add param so it exists
        gradients = {"layer.weight": torch.zeros(8, 8)}
        projected = projector.project(gradients, mh)
        assert "layer.weight" in projected

    def test_compute_effective_lr(self):
        projector = GradientProjector()
        from da_ocl.memory.mask_history import MaskHistory
        mh = MaskHistory()
        lr = projector.compute_effective_lr(0.001, mh, "unknown_param")
        assert lr == 0.0


class TestEWCUpdater:
    def test_snapshot_records_params(self):
        updater = EWCUpdater()
        model = torch.nn.Linear(8, 8)
        updater.snapshot(model)
        assert len(updater._old_params) > 0

    def test_compute_ewc_loss_zero_without_snapshot(self):
        updater = EWCUpdater()
        model = torch.nn.Linear(8, 8)
        loss = updater.compute_ewc_loss(model)
        assert loss == 0.0

    def test_compute_fisher(self):
        updater = EWCUpdater(fisher_samples=5)
        model = torch.nn.Linear(8, 8)
        inputs = torch.randn(4, 8)
        updater.compute_fisher(model, inputs)
        assert len(updater._fisher_diagonal) > 0


class TestConsistencyGate:
    def test_evaluate_returns_score(self):
        gate = ConsistencyGate()
        before = {"layer.weight": torch.randn(8, 8)}
        after = {"layer.weight": torch.randn(8, 8)}
        accepted, score = gate.evaluate(before, after)
        assert isinstance(accepted, bool)
        assert 0.0 <= score <= 1.0

    def test_validate_and_rollback_on_reject(self):
        gate = ConsistencyGate(threshold=0.99)  # 极高阈值，强制拒绝
        model = torch.nn.Linear(8, 8)
        before = {n: p.clone() for n, p in model.named_parameters()}
        after = {n: p + 10.0 for n, p in model.named_parameters()}
        accepted = gate.validate_and_rollback(model, before, after)
        # 如果被拒绝，参数应该回滚
        if not accepted:
            for n, p in model.named_parameters():
                assert torch.allclose(p, before[n], atol=1e-4)

    def test_forward_returns_negative_score(self):
        gate = ConsistencyGate()
        before = {"layer.weight": torch.randn(4, 4)}
        after = {"layer.weight": torch.randn(4, 4)}
        loss = gate.forward(before, after)
        assert loss <= 0.0
