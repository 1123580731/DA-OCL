"""
DA-OCL 配置加载模块
所有超参数的单一来源，由 OmegaConf 从 YAML 加载。
禁止在业务代码中硬编码超参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """模型架构配置"""
    obs_dim: int = 128
    entity_dim: int = 128
    gnn_hidden_dim: int = 256
    ssm_state_dim: int = 512
    rule_dim: int = 256
    action_dim: int = 32
    window_size: int = 6
    use_mamba: bool = True


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    fast_memory_capacity: int = 5000
    slow_memory_lr: float = 0.001
    use_mamba: bool = True


@dataclass
class BufferConfig:
    """Buffer 配置"""
    total_capacity: int = 10000
    protected_ratio: float = 0.10
    diversity_ratio_initial: float = 0.40
    diversity_ratio_mature: float = 0.20
    task_ratio_initial: float = 0.50
    task_ratio_mature: float = 0.70


@dataclass
class RegulationConfig:
    """三条调控轴配置"""
    alpha_importance: float = 0.7
    surprise_sigma: float = 1.0
    lambda_decay: float = 0.01
    death_r_penalty_threshold: float = 8.0
    death_causal_chain_T: int = 8
    death_priority_gamma: float = 0.9


@dataclass
class NightConfig:
    """夜晚阶段配置"""
    window_threshold: int = 500
    mask_density_threshold: float = 0.3
    surp_threshold: float = 2.0
    timestep_fallback: int = 5000
    beta0_kl: float = 1.0
    gamma_dream: float = 0.1
    ewc_lambda: float = 5000.0


@dataclass
class FilterConfig:
    """对抗滤波配置"""
    coarse_threshold: float = 0.5
    fine_threshold: float = 0.7
    skip_fine_if_coarse_above: float = 0.5


@dataclass
class BoundaryConfig:
    """情景边界检测配置"""
    n_sigma: float = 2.0
    history_size: int = 50


@dataclass
class EnvConfig:
    """环境配置"""
    name: str = "dummy"
    num_entities: int = 4
    obs_dim: int = 128
    action_dim: int = 32
    max_episode_steps: int = 1000


@dataclass
class TrainConfig:
    """训练配置"""
    num_steps: int = 100000
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 100
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"


@dataclass
class DAOCLConfig:
    """DA-OCL 总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    regulation: RegulationConfig = field(default_factory=RegulationConfig)
    night: NightConfig = field(default_factory=NightConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(yaml_path: str | Path | None = None) -> DAOCLConfig:
    """
    从 YAML 文件加载配置。如未指定路径，加载默认配置。

    Args:
        yaml_path: 配置文件路径，若为 None 则使用所有默认值

    Returns:
        DAOCLConfig 实例

    Raises:
        FileNotFoundError: yaml_path 指定但文件不存在
        ValueError: YAML 格式无效
    """
    if yaml_path is None:
        return DAOCLConfig()

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    cfg_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    assert isinstance(cfg_dict, dict)

    def _resolve(cfg: dict) -> dict:
        """递归处理嵌套字典，确保所有值可序列化"""
        result = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                result[k] = _resolve(v)
            else:
                result[k] = v
        return result

    cfg = _resolve(cfg_dict)

    # 映射到 dataclass
    return DAOCLConfig(
        model=ModelConfig(**cfg.get("model", {})),
        memory=MemoryConfig(**cfg.get("memory", {})),
        buffer=BufferConfig(**cfg.get("buffer", {})),
        regulation=RegulationConfig(**cfg.get("regulation", {})),
        night=NightConfig(**cfg.get("night", {})),
        filter=FilterConfig(**cfg.get("filter", {})),
        boundary=BoundaryConfig(**cfg.get("boundary", {})),
        env=EnvConfig(**cfg.get("env", {})),
        train=TrainConfig(**cfg.get("train", {})),
    )


def config_to_omegaconf(cfg: DAOCLConfig) -> DictConfig:
    """将 DAOCLConfig 转换为 OmegaConf DictConfig（用于 Hydra-style 访问）"""
    return OmegaConf.structured(cfg)


# 全局默认配置实例（延迟初始化）
_default_config: DAOCLConfig | None = None


def get_default_config() -> DAOCLConfig:
    """获取全局默认配置（单例）"""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config
