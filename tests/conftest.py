"""
pytest fixtures for DA-OCL tests
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch

# 将项目根目录加入 sys.path，使 envs/ 包可被导入
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from da_ocl.config import DAOCLConfig, get_default_config, load_config
from da_ocl.core.constants import (
    ACTION_DIM,
    COMPOSITE_DIM,
    ENTITY_DIM,
    GNN_HIDDEN_DIM,
    OBS_DIM,
    RULE_DIM,
    SSM_STATE_DIM,
    WINDOW_SIZE_DEFAULT,
)


@pytest.fixture(scope="session")
def default_config() -> DAOCLConfig:
    """全局默认配置 fixture"""
    return get_default_config()


@pytest.fixture(scope="session")
def debug_config() -> DAOCLConfig:
    """调试配置 fixture（小规模参数）"""
    cfg = get_default_config()
    # 覆盖为调试规模
    cfg.model.obs_dim = OBS_DIM
    cfg.model.entity_dim = ENTITY_DIM
    cfg.model.gnn_hidden_dim = GNN_HIDDEN_DIM
    cfg.model.ssm_state_dim = SSM_STATE_DIM
    cfg.model.rule_dim = RULE_DIM
    cfg.model.action_dim = 8
    cfg.model.window_size = 5
    cfg.model.use_mamba = False
    cfg.memory.use_mamba = False
    cfg.buffer.total_capacity = 500
    cfg.train.device = "cpu"
    return cfg


@pytest.fixture
def torch_device() -> torch.device:
    """当前可用的 PyTorch 设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed() -> Generator[None, None, None]:
    """为每个测试设置随机种子的 fixture"""
    seed = random.randint(0, 2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    yield
    # 测试后不恢复种子（测试间隔离）


@pytest.fixture
def sample_obs(torch_device: torch.device) -> torch.Tensor:
    """随机单帧观测 fixture"""
    return torch.randn(OBS_DIM, device=torch_device)


@pytest.fixture
def sample_batch_obs(batch_size: int = 4, torch_device: torch.device = None) -> torch.Tensor:
    """随机批次观测 fixture（如果未指定 torch_device 则使用 fixture）"""
    return torch.randn(batch_size, OBS_DIM, device=torch_device or torch.device("cpu"))


@pytest.fixture
def sample_entity_window(torch_device: torch.device) -> torch.Tensor:
    """随机因果窗口 fixture (T, entity_dim)"""
    return torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device)


@pytest.fixture
def sample_composite(torch_device: torch.device) -> torch.Tensor:
    """随机复合特征 fixture (COMPOSITE_DIM,)"""
    return torch.randn(COMPOSITE_DIM, device=torch_device)


@pytest.fixture
def sample_composite_batch(batch_size: int = 4, torch_device: torch.device = None) -> torch.Tensor:
    """随机复合特征批次 fixture"""
    return torch.randn(batch_size, COMPOSITE_DIM, device=torch_device or torch.device("cpu"))
