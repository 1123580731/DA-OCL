"""
DA-OCL: 双驱非对称在线持续学习框架
Dual-Archetype Asymmetric Online Continual Learning

版本: v2.0
架构: 因果窗口输入 · 联合掩码沉默写入 · 三区海马体Buffer · 夜晚沉默痕迹激活
"""

from __future__ import annotations

__version__ = "2.0.0"

from da_ocl.config import (
    DAOCLConfig,
    get_default_config,
    load_config,
)
from da_ocl.core.constants import (
    COMPOSITE_DIM,
    ENTITY_DIM,
    GNN_HIDDEN_DIM,
    RULE_DIM,
    SSM_STATE_DIM,
    WINDOW_SIZE_DEFAULT,
    OBS_DIM,
    ACTION_DIM,
)
from da_ocl.agent import DAOCLAgent
from da_ocl.buffer.experience import Experience
from da_ocl.memory.fast_memory import FastMemory
from da_ocl.memory.slow_memory import SlowMemory
from da_ocl.memory.mask_history import MaskHistory
from da_ocl.memory.joint_mask import JointMaskComputer
from da_ocl.causal.window_manager import CausalWindowManager
from da_ocl.causal.surprise_detector import SurpriseDetector
from da_ocl.causal.boundary_detector import BoundaryDetector
from da_ocl.night.night_trigger import NightTrigger
from da_ocl.night.night_pipeline import NightPipeline

__all__ = [
    "__version__",
    # 配置
    "DAOCLConfig",
    "get_default_config",
    "load_config",
    # 维度常量
    "SSM_STATE_DIM",
    "COMPOSITE_DIM",
    "ENTITY_DIM",
    "GNN_HIDDEN_DIM",
    "RULE_DIM",
    "OBS_DIM",
    "ACTION_DIM",
    "WINDOW_SIZE_DEFAULT",
    # 核心智能体
    "DAOCLAgent",
    # 经验
    "Experience",
    # 记忆
    "FastMemory",
    "SlowMemory",
    "MaskHistory",
    "JointMaskComputer",
    # 因果
    "CausalWindowManager",
    "SurpriseDetector",
    "BoundaryDetector",
    # 夜晚
    "NightTrigger",
    "NightPipeline",
]
