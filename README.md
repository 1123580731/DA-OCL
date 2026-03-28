# DA-OCL

**双驱非对称在线持续学习框架**（Dual-Stream Asymmetric Online Continual Learning）

解决高随机性开放世界环境（Minecraft 等）中智能体的持续学习问题。核心哲学：**不预测世界的样子，而是理解世界运行的规律**。

---

## 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0
- 可选：`mamba-ssm`（如安装则自动使用 Mamba SSM 加速，否则自动降级到 PyTorch GRU 实现）

## 安装

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 开发依赖（含 pytest）

# 可选：安装 mamba-ssm（需 CUDA）
pip install causal-conv1d mamba-ssm
```

## 快速开始

```bash
# 使用 Dummy 环境训练 1000 步（默认，无需外部依赖）
python experiments/run_experiment.py

# 使用 YAML 配置
python experiments/run_experiment.py --config experiments/configs/config.yaml
```

## 架构概述

```
白天阶段（在线学习）
  观测 → 编码器 → 因果窗口管理器
                   ├→ 慢记忆检索 → 规律激活向量 o_hat
                   └→ 联合掩码计算 → 沉默写入
  汇合 → 对抗滤波 → 双指标计算 → 三区 Buffer 写入

夜晚阶段（离线整合）
  掩码历史索引 → 蒸馏损失 + EWC 正则化 + 梦境损失
              → 梯度投影 → 一致性验证门控
              → 慢记忆参数更新（通过）或回滚（拒绝）
```

**三区海马体启发 Buffer：**
- **保护区**（10%）：死亡经验 + 因果链追溯，不参与正常竞争替换
- **规律多样性区**（20~40%）：保护稀疏规律域的代表性经验
- **任务导向区**（50~70%）：对当前任务目标有价值的经验，支持动态重排序

## 目录结构

```
DA-OCL/
├── da_ocl/                  # 核心框架
│   ├── core/                # 编码器、GNN、SSM、规则约束
│   ├── causal/              # 因果窗口、惊喜度检测、边界检测
│   ├── memory/              # 快记忆（MHN）、慢记忆（3层）、联合掩码
│   ├── day/                 # 白天流程：滤波、指标、检索、预写入
│   ├── buffer/              # 三区 Buffer + 双缓冲
│   ├── night/               # 夜晚流程：蒸馏、梦境、梯度、EWC
│   ├── losses/              # 损失函数集合
│   ├── regulation/          # 三条调控轴（重要性/惊喜度/时间衰减）
│   └── agent.py             # DAOCLAgent 主类
├── envs/                    # 环境接口
│   ├── base_env.py          # 环境基类
│   └── dummy_env.py         # 测试用 Dummy 环境
├── experiments/             # 实验入口
│   └── run_experiment.py
├── tests/                   # 测试套件（pytest）
│   ├── test_m1_skeleton.py
│   ├── test_m2_encoder.py
│   ├── test_m3_memory.py
│   ├── test_m4_day_phase.py
│   ├── test_m5_buffer.py
│   ├── test_m6_night_phase.py
│   ├── test_m7_integration.py
│   └── conftest.py
├── da_ocl/config.py         # OmegaConf 配置管理
└── da_ocl/core/constants.py # 维度契约常量
```

## 关键维度（Dimension Contract）

| 符号 | 值 | 说明 |
|------|-----|------|
| `obs_dim` | 128 | 单帧观测维度 |
| `entity_dim` | 128 | 实体嵌入维度 |
| `gnn_hidden_dim` | 256 | GNN 中间层维度 |
| `ssm_state_dim` | **512** | **全系统唯一基准** |
| `composite_dim` | **1536** | **= 3 × ssm_state_dim** |
| `window_size` | 6 | 因果窗口长度（认知参数） |

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定里程碑测试
pytest tests/test_m7_integration.py -v

# 带覆盖率
pytest tests/ --cov=da_ocl --cov-report=html
```

当前状态：**139 个测试全部通过**（M1~M7 全部 milestone 验收完成）

## 扩展到真实环境

将 `envs/dummy_env.py` 替换为真实环境接口（继承 `BaseEnv`），参考 `envs/base_env.py` 中的 `EnvOutput` dataclass 提供 `obs`、`reward`、`done` 等字段。

## 核心论文参考

- **记忆形成**：Kitamura et al., Science, 2017（海马体-新皮层记忆双痕迹）
- **慢记忆 SSM**：Mamba-SSM（Gu & Dao, 2023），失败时自动降级为 PyTorch GRU

## License

MIT
