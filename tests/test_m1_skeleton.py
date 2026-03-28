"""
M1 骨架测试：验证包导入和维度契约
"""

from __future__ import annotations

import torch

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


class TestConstantsIntegrity:
    """验证维度契约的一致性和正确性"""

    def test_composite_dim_matches_formula(self):
        """COMPOSITE_DIM 必须等于 3 * SSM_STATE_DIM"""
        assert COMPOSITE_DIM == 3 * SSM_STATE_DIM
        assert COMPOSITE_DIM == 1536, f"COMPOSITE_DIM 应为 1536，实际为 {COMPOSITE_DIM}"

    def test_entity_dim_reasonable(self):
        """ENTITY_DIM 应为合理正值"""
        assert ENTITY_DIM > 0
        assert isinstance(ENTITY_DIM, int)

    def test_ssm_state_dim_is_reference(self):
        """SSM_STATE_DIM 是全系统基准，应为最大值"""
        assert SSM_STATE_DIM >= GNN_HIDDEN_DIM
        assert SSM_STATE_DIM >= RULE_DIM
        assert SSM_STATE_DIM >= ENTITY_DIM * 2

    def test_window_size_reasonable(self):
        """窗口长度应在 5~8 范围内（认知有效区间）"""
        assert 5 <= WINDOW_SIZE_DEFAULT <= 8


class TestPackageImports:
    """验证包可正确导入"""

    def test_core_imports(self):
        """core 模块可导入"""
        from da_ocl.core import constants

        assert constants.OBS_DIM == OBS_DIM
        assert constants.SSM_STATE_DIM == SSM_STATE_DIM
        assert constants.COMPOSITE_DIM == COMPOSITE_DIM

    def test_config_imports(self):
        """配置模块可导入"""
        from da_ocl.config import (
            DAOCLConfig,
            FilterConfig,
            get_default_config,
            load_config,
            ModelConfig,
            NightConfig,
            RegulationConfig,
            TrainConfig,
        )

        cfg = get_default_config()
        assert isinstance(cfg, DAOCLConfig)
        assert cfg.model.obs_dim == OBS_DIM
        assert cfg.model.ssm_state_dim == SSM_STATE_DIM

    def test_config_defaults_valid(self, debug_config):
        """调试配置应覆盖关键参数"""
        assert debug_config.model.window_size <= 6
        assert debug_config.train.device == "cpu"

    def test_env_base_class_import(self):
        """环境基类可导入"""
        from envs import BaseEnv, EnvOutput

        assert BaseEnv is not None
        assert EnvOutput is not None

    def test_load_config_from_yaml(self):
        """可以从 YAML 加载配置（用 io.StringIO 避免文件 I/O）"""
        import io
        from omegaconf import OmegaConf
        from da_ocl.config import (
            DAOCLConfig, ModelConfig, MemoryConfig,
            BufferConfig, TrainConfig,
        )

        yaml_content = """
model:
  obs_dim: 128
  entity_dim: 128
  gnn_hidden_dim: 256
  ssm_state_dim: 512
  rule_dim: 256
  action_dim: 16
  window_size: 7
  use_mamba: false
memory:
  fast_memory_capacity: 1000
  use_mamba: false
buffer:
  total_capacity: 2000
train:
  num_steps: 1000
  device: cpu
  seed: 0
"""
        cfg_dict = OmegaConf.to_container(OmegaConf.load(io.StringIO(yaml_content)), resolve=True)

        def _resolve(cfg):
            result = {}
            for k, v in cfg.items():
                if isinstance(v, dict):
                    result[k] = _resolve(v)
                else:
                    result[k] = v
            return result

        cfg = DAOCLConfig(
            model=ModelConfig(**_resolve(cfg_dict).get("model", {})),
            memory=MemoryConfig(**_resolve(cfg_dict).get("memory", {})),
            buffer=BufferConfig(**_resolve(cfg_dict).get("buffer", {})),
            train=TrainConfig(**_resolve(cfg_dict).get("train", {})),
        )
        assert cfg.model.window_size == 7
        assert cfg.model.action_dim == 16
        assert cfg.memory.fast_memory_capacity == 1000

    def test_load_config_file_not_found(self):
        """加载不存在的配置文件应抛出 FileNotFoundError"""
        from da_ocl.config import load_config

        try:
            load_config("/nonexistent/path/config.yaml")
            raise AssertionError("应抛出 FileNotFoundError")
        except FileNotFoundError:
            pass


class TestConstantsUsage:
    """验证常量可被业务代码正确使用"""

    def test_tensor_creation_with_constants(self, torch_device):
        """使用常量创建张量应在 GPU/CPU 上都有效"""
        obs = torch.randn(OBS_DIM, device=torch_device)
        assert obs.shape == (OBS_DIM,)

        entity = torch.randn(ENTITY_DIM, device=torch_device)
        assert entity.shape == (ENTITY_DIM,)

        ssm_state = torch.randn(SSM_STATE_DIM, device=torch_device)
        assert ssm_state.shape == (SSM_STATE_DIM,)

        composite = torch.randn(COMPOSITE_DIM, device=torch_device)
        assert composite.shape == (COMPOSITE_DIM,)

        window = torch.randn(WINDOW_SIZE_DEFAULT, ENTITY_DIM, device=torch_device)
        assert window.shape == (WINDOW_SIZE_DEFAULT, ENTITY_DIM)

    def test_constants_are_final(self):
        """验证常量是不可变的（通过尝试修改）"""
        # 注意：Final 仅在类型检查时强制，运行时 Python 无法真正防止修改
        # 此测试验证当前值正确
        assert OBS_DIM == 128
        assert ENTITY_DIM == 128
        assert GNN_HIDDEN_DIM == 256
        assert SSM_STATE_DIM == 512
        assert COMPOSITE_DIM == 1536
        assert RULE_DIM == 256
        assert ACTION_DIM == 32
