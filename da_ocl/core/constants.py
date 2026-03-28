"""
DA-OCL v2 维度契约
所有模块共享的唯一真实来源（Single Source of Truth）
禁止在业务代码中硬编码维度数字，必须从此模块导入。
"""

from __future__ import annotations

from typing import Final

# ============================================================
# 维度参数
# ============================================================

OBS_DIM: Final[int] = 128
"""单帧原始环境观测维度"""

ENTITY_DIM: Final[int] = 128
"""实体嵌入维度（编码器输出）"""

GNN_HIDDEN_DIM: Final[int] = 256
"""GNN 中间层维度"""

SSM_STATE_DIM: Final[int] = 512
"""SSM 隐状态维度，全系统唯一基准"""

RULE_DIM: Final[int] = 256
"""元规则向量维度（逻辑约束层内部）"""

ACTION_DIM: Final[int] = 32
"""动作向量维度（由环境决定）"""

COMPOSITE_DIM: Final[int] = 3 * SSM_STATE_DIM  # = 1536
"""因果窗口级别复合特征 B 的维度 = 3 × ssm_state_dim"""

# ============================================================
# 因果窗口参数
# ============================================================

WINDOW_SIZE_MIN: Final[int] = 5
"""因果窗口最小长度"""

WINDOW_SIZE_MAX: Final[int] = 8
"""因果窗口最大长度"""

WINDOW_SIZE_DEFAULT: Final[int] = 6
"""默认因果窗口长度（认知上有意义的最小完整单元）"""

# ============================================================
# Buffer 区域容量比例
# ============================================================

BUFFER_ZONE_RATIOS_INITIAL: Final[dict[str, float]] = {
    "protected": 0.10,
    "diversity": 0.40,
    "task": 0.50,
}
"""初期三区容量分配比例"""

BUFFER_ZONE_RATIOS_MATURE: Final[dict[str, float]] = {
    "protected": 0.10,  # 固定
    "diversity": 0.20,  # 萎缩
    "task": 0.70,       # 扩张
}
"""成熟期三区容量分配比例"""

# ============================================================
# 调控轴参数默认值
# ============================================================

SURPRISE_SIGMA: Final[float] = 1.0
"""惊喜度归一化系数"""

LAMBDA_DECAY: Final[float] = 0.01
"""Delta_t 时间衰减系数"""

ALPHA_IMPORTANCE: Final[float] = 0.7
"""Importance 中 (1-CoverageRate) 的权重"""

IMPORTANCE_MEAN_INIT: Final[float] = 0.5
"""重要性调控初始均值（概率空间）"""

IMPORTANCE_DECAY_RATE: Final[float] = 0.1
"""重要性衰减率"""

SURPRISE_MEAN_INIT: Final[float] = 0.3
"""惊喜度调控初始均值（概率空间）"""

SURPRISE_DECAY_RATE: Final[float] = 0.1
"""惊喜度衰减率"""

TEMPORAL_DECAY_COEFFICIENT: Final[float] = 0.05
"""时间衰减系数"""

REGULATION_DIM: Final[int] = 128
"""调控向量维度"""

SLOW_MEMORY_DIM: Final[int] = SSM_STATE_DIM
"""慢记忆参数维度（等于 SSM_STATE_DIM = 512，作为各模块维度契约基准）"""

BETA0_KL: Final[float] = 1.0
"""KL 损失基准系数"""

ALPHA_KL: Final[float] = 0.5
"""KL 中 Importance 的衰减系数"""

GAMMA_DREAM: Final[float] = 0.1
"""梦境损失权重系数"""

EWC_LAMBDA: Final[float] = 5000.0
"""EWC 正则化强度"""

# ============================================================
# 情景边界检测参数
# ============================================================

BOUNDARY_N_SIGMA: Final[float] = 2.0
"""边界触发倍数（mu_history + N * sigma_history）"""

BOUNDARY_HISTORY_SIZE: Final[int] = 50
"""计算历史均值/方差的历史窗口大小"""

# ============================================================
# 夜晚触发条件
# ============================================================

NIGHT_WINDOW_THRESHOLD: Final[int] = 500
"""Buffer 窗口数量阈值"""

NIGHT_MASK_DENSITY_THRESHOLD: Final[float] = 0.3
"""掩码激活密度阈值"""

NIGHT_SURP_THRESHOLD: Final[float] = 2.0
"""滑动窗口平均 Surp 阈值"""

NIGHT_TIMESTEP_FALLBACK: Final[int] = 5000
"""时间步数兜底触发"""

NIGHT_EPOCHS: Final[int] = 3
"""夜晚优化轮数"""

DISTILL_LR: Final[float] = 1e-4
"""蒸馏学习率"""

DREAM_NUM_SAMPLES: Final[int] = 32
"""梦境样本数量"""

DREAM_VARIANCE_INIT: Final[float] = 0.1
"""梦境生成方差初始值"""

EWC_FISHER_SAMPLES: Final[int] = 10
"""Fisher 信息估计样本数"""

GRADIENT_PROJECTION_EPS: Final[float] = 1e-6
"""梯度投影阈值"""

CONSISTENCY_GATE_THRESHOLD: Final[float] = 0.5
"""一致性门控阈值"""

# ============================================================
# 死亡经验参数
# ============================================================

DEATH_R_PENALTY_THRESHOLD: Final[float] = 8.0
"""触发保护区的 r_penalty 绝对阈值"""

DEATH_CAUSAL_CHAIN_T: Final[int] = 8
"""死亡前因果链追溯帧数"""

DEATH_PRIORITY_GAMMA: Final[float] = 0.9
"""死亡追溯优先级衰减系数"""

# ============================================================
# 对抗滤波参数
# ============================================================

FILTER_COARSE_THRESHOLD: Final[float] = 0.5
"""第一级粗筛阈值（batch 整体）"""

FILTER_FINE_THRESHOLD: Final[float] = 0.7
"""第二级细筛阈值（逐帧）"""

FILTER_SKIP_FINE_IF_COARSE_ABOVE: Final[float] = 0.5
"""粗筛过滤比超过此值时跳过细筛"""

# ============================================================
# MHN 快记忆参数
# ============================================================

MHN_BETA: Final[float] = 1.0
"""MHN 能量函数的温度参数"""

MHN_RETRIEVE_ITERS: Final[int] = 10
"""MHN 联想检索的迭代次数"""

# ============================================================
# EWC 参数
# ============================================================

EWC_TRAIN_ITERS: Final[int] = 100
"""计算 Fisher 信息时的训练迭代次数"""

# ============================================================
# 验证常量（用于编译时检查）
# ============================================================

__all__ = [
    # 维度
    "OBS_DIM",
    "ENTITY_DIM",
    "GNN_HIDDEN_DIM",
    "SSM_STATE_DIM",
    "RULE_DIM",
    "ACTION_DIM",
    "COMPOSITE_DIM",
    # 窗口
    "WINDOW_SIZE_MIN",
    "WINDOW_SIZE_MAX",
    "WINDOW_SIZE_DEFAULT",
    # Buffer
    "BUFFER_ZONE_RATIOS_INITIAL",
    "BUFFER_ZONE_RATIOS_MATURE",
    # 调控轴
    "SURPRISE_SIGMA",
    "LAMBDA_DECAY",
    "ALPHA_IMPORTANCE",
    "IMPORTANCE_MEAN_INIT",
    "IMPORTANCE_DECAY_RATE",
    "SURPRISE_MEAN_INIT",
    "SURPRISE_DECAY_RATE",
    "TEMPORAL_DECAY_COEFFICIENT",
    "REGULATION_DIM",
    "SLOW_MEMORY_DIM",
    "BETA0_KL",
    "ALPHA_KL",
    "GAMMA_DREAM",
    "EWC_LAMBDA",
    # 边界
    "BOUNDARY_N_SIGMA",
    "BOUNDARY_HISTORY_SIZE",
    # 夜晚
    "NIGHT_WINDOW_THRESHOLD",
    "NIGHT_MASK_DENSITY_THRESHOLD",
    "NIGHT_SURP_THRESHOLD",
    "NIGHT_TIMESTEP_FALLBACK",
    "NIGHT_EPOCHS",
    "DISTILL_LR",
    "DREAM_NUM_SAMPLES",
    "DREAM_VARIANCE_INIT",
    "EWC_FISHER_SAMPLES",
    "GRADIENT_PROJECTION_EPS",
    "CONSISTENCY_GATE_THRESHOLD",
    # 死亡
    "DEATH_R_PENALTY_THRESHOLD",
    "DEATH_CAUSAL_CHAIN_T",
    "DEATH_PRIORITY_GAMMA",
    # 滤波
    "FILTER_COARSE_THRESHOLD",
    "FILTER_FINE_THRESHOLD",
    "FILTER_SKIP_FINE_IF_COARSE_ABOVE",
    # MHN
    "MHN_BETA",
    "MHN_RETRIEVE_ITERS",
    # EWC
    "EWC_TRAIN_ITERS",
]
