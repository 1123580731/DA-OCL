"""
DA-OCL Night 模块
夜晚流程：蒸馏、梦境生成、梯度投影、EWC、Consistency Gate
"""

from __future__ import annotations

from da_ocl.night.night_trigger import NightTrigger
from da_ocl.night.phase_controller import NightPhaseController
from da_ocl.night.night_pipeline import NightPipeline
from da_ocl.night.distillation import DistillationPath
from da_ocl.night.dream_generator import DreamGenerator
from da_ocl.night.gradient_projector import GradientProjector
from da_ocl.night.ewc_updater import EWCUpdater
from da_ocl.night.consistency_gate import ConsistencyGate

__all__ = [
    "NightTrigger",
    "NightPhaseController",
    "NightPipeline",
    "DistillationPath",
    "DreamGenerator",
    "GradientProjector",
    "EWCUpdater",
    "ConsistencyGate",
]
