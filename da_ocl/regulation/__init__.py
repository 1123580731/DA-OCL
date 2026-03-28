"""
DA-OCL Regulation 模块
调节轴：重要性、惊喜度、时间衰减
"""

from __future__ import annotations

from da_ocl.regulation.axes import (
    ImportanceRegulation,
    SurpriseRegulation,
    TemporalDecayRegulation,
    RegulationMixer,
)

__all__ = [
    "ImportanceRegulation",
    "SurpriseRegulation",
    "TemporalDecayRegulation",
    "RegulationMixer",
]
