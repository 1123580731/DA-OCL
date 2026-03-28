"""
DA-OCL 核心模块
Constants + Encoder + GNN + SSM + RuleConstraints + CompositeBuilder
"""

from __future__ import annotations

from da_ocl.core.constants import *
from da_ocl.core.encoder import Encoder
from da_ocl.core.gnn_layer import GNNRelationEncoder
from da_ocl.core.ssm_layer import SSMCompressor
from da_ocl.core.rule_constraints import RuleConstraintLayer
from da_ocl.core.composite_builder import CompositeFeatureBuilder

__all__ = [
    # Constants
    "OBS_DIM",
    "ENTITY_DIM",
    "GNN_HIDDEN_DIM",
    "SSM_STATE_DIM",
    "RULE_DIM",
    "ACTION_DIM",
    "COMPOSITE_DIM",
    # Core modules
    "Encoder",
    "GNNRelationEncoder",
    "SSMCompressor",
    "RuleConstraintLayer",
    "CompositeFeatureBuilder",
]
