# DA-OCL 技术规范（SPEC）

> 本文档是 `PROJECT.md` 的补充，记录所有模块的**接口契约、数据格式、异常行为**。随代码开发逐步填充。

---

## 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-03-28 | 初始骨架（待开发填充） |

---

## 模块间接口总览

```python
# === 数据类型别名 ===
EntityWindow = List[Tensor]           # T 个 (entity_dim,) 张量的列表
CompositeWindow = List[Tensor]        # T 个 (COMPOSITE_DIM,) 张量的列表
Experience = dataclasses dataclass    # 见 §6.10.2
MaskTensor = Tensor                   # 与慢记忆参数同 shape，值为标量广播
SlowMemoryState = Dict[str, Tensor]  # torch.state_dict() 格式
BufferState = Dict                    # 三区 Buffer 的序列化格式

# === 阶段返回值 ===
DayMetrics = {
    "surprise": float,
    "importance": float,
    "coverage_rate": float,
    "mask_value": float,
    "filter_passed": bool,
    "new_boundary": bool,
    "added_to_buffer": bool,
}

NightMetrics = {
    "l_total": float,
    "l_distill": float,
    "l_kl": float,
    "l_ewc": float,
    "l_dream": float,
    "buffer_sampled": int,
    "dream_sampled": int,
    "params_updated": int,
    "consistency_passed": bool,
}
```

---

## da_ocl/core/constants.py

```python
# ---------- 维度 ----------
OBS_DIM: Final[int] = 128
ENTITY_DIM: Final[int] = 128
GNN_HIDDEN_DIM: Final[int] = 256
SSM_STATE_DIM: Final[int] = 512       # 全系统唯一基准维度
RULE_DIM: Final[int] = 256
ACTION_DIM: Final[int] = 32
COMPOSITE_DIM: Final[int] = 1536       # = 3 * SSM_STATE_DIM

# ---------- 因果窗口 ----------
WINDOW_SIZE_MIN: Final[int] = 5
WINDOW_SIZE_MAX: Final[int] = 8
WINDOW_SIZE_DEFAULT: Final[int] = 6

# ---------- 三区 Buffer ----------
PROTECTED_RATIO: Final[float] = 0.10
DIVERSITY_RATIO_INIT: Final[float] = 0.40
DIVERSITY_RATIO_MATURE: Final[float] = 0.20
TASK_RATIO_INIT: Final[float] = 0.50
TASK_RATIO_MATURE: Final[float] = 0.70

# ---------- 调控轴 ----------
SURPRISE_SIGMA: Final[float] = 1.0
LAMBDA_DECAY: Final[float] = 0.01
ALPHA_IMPORTANCE: Final[float] = 0.7
BETA0_KL: Final[float] = 1.0
ALPHA_KL: Final[float] = 0.5
GAMMA_DREAM: Final[float] = 0.1
EWC_LAMBDA: Final[float] = 5000.0

# ---------- 边界检测 ----------
BOUNDARY_N_SIGMA: Final[float] = 2.0
BOUNDARY_HISTORY_SIZE: Final[int] = 50

# ---------- 夜晚触发 ----------
NIGHT_WINDOW_THRESHOLD: Final[int] = 500
NIGHT_MASK_DENSITY_THRESHOLD: Final[float] = 0.3
NIGHT_SURP_THRESHOLD: Final[float] = 2.0
NIGHT_TIMESTEP_FALLBACK: Final[int] = 5000

# ---------- 死亡经验 ----------
DEATH_R_PENALTY_THRESHOLD: Final[float] = 8.0
DEATH_CAUSAL_CHAIN_T: Final[int] = 8
DEATH_PRIORITY_GAMMA: Final[float] = 0.9
```

---

## da_ocl/agent.py

```python
class DAOCLAgent(ABC):

    def __init__(self, config: OmegaConf.DictConfig) -> None:
        """
        初始化所有子系统：
        编码器、因果窗口管理器、双记忆、三区Buffer、夜晚调度器、双缓冲管理器
        """

    def act(self, obs: Tensor) -> int:
        """
        推理接口（白天）。

        Args:
            obs: (obs_dim,) 单帧观测张量

        Returns:
            action: int，动作索引

        Raises:
            RuntimeError: 慢记忆未初始化时调用
        """
        ...

    def update(self, obs: Tensor, action: int, reward: float, done: bool) -> DayMetrics:
        """
        训练接口（白天单步）。

        Args:
            obs: (obs_dim,) 当前帧观测
            action: int，执行的动作
            reward: float，奖励信号
            done: bool， episode 是否结束

        Returns:
            DayMetrics 字典，包含 surprise、importance、coverage_rate 等

        Raises:
            ValueError: obs shape 不匹配 obs_dim
        """
        ...

    def should_trigger_night(self) -> bool:
        """
        夜晚触发判断（外部调度器调用）。

        Returns:
            True if any trigger condition is met
        """
        ...

    def night_phase(self) -> NightMetrics:
        """
        夜晚整合（外部调度器触发后调用）。

        Returns:
            NightMetrics 字典，包含各损失分量和一致性验证结果

        Raises:
            RuntimeError: 掩码历史为空时调用（应先检查 should_trigger_night）
        """
        ...

    @property
    def buffer(self) -> ThreeZoneBuffer: ...

    @property
    def slow_memory(self) -> SlowMemory: ...

    @property
    def fast_memory(self) -> FastMemory: ...

    def save(self, path: Path | str) -> None:
        """
        保存完整 checkpoint。

        Args:
            path: 保存路径 (.pt 文件)

        Raises:
            OSError: 路径不可写
        """
        ...

    def load(self, path: Path | str) -> None:
        """
        加载 checkpoint。

        Args:
            path: checkpoint 路径

        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: checkpoint 版本不匹配
        """
        ...
```

---

## da_ocl/core/encoder.py

```python
class Encoder(nn.Module):
    obs_dim: ClassVar[int] = OBS_DIM
    entity_dim: ClassVar[int] = ENTITY_DIM

    def __init__(self) -> None:
        super().__init__()
        ...

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: (obs_dim,) 或 (batch, obs_dim)

        Returns:
            entity: (entity_dim,) 或 (batch, entity_dim)

        Raises:
            RuntimeError: obs dim 不匹配

        Tested-by: test_encoder.py::test_encoder_forward_shape
        """
        ...

    def get_output_dim(self) -> int:
        """返回 ENTITY_DIM = 128（编译时验证）"""
        return ENTITY_DIM
```

---

## da_ocl/core/gnn_layer.py

```python
class GNNRelationEncoder(nn.Module):
    gnn_hidden_dim: ClassVar[int] = GNN_HIDDEN_DIM

    def __init__(self, num_entities: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_heads = num_heads

    def forward(
        self,
        entity_window: Tensor,
        entity_positions: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            entity_window: (T, entity_dim) 窗口内实体的嵌入序列
            entity_positions: (T, num_entities, 2) 每帧的实体位置（可选）

        Returns:
            relational_features: (T, GNN_HIDDEN_DIM)
                              跨帧实体关系表征

        Raises:
            RuntimeError: entity_window 的 dim 不匹配

        Tested-by: test_encoder.py::test_gnn_forward
        """
        ...
```

---

## da_ocl/core/ssm_layer.py

```python
class SSMCompressor(nn.Module):
    ssm_state_dim: ClassVar[int] = SSM_STATE_DIM

    def __init__(self, use_mamba: bool = True) -> None:
        """
        Args:
            use_mamba: 是否优先使用 mamba-ssm，失败则自动降级

        Raises:
            ImportError: use_mamba=True 但 mamba-ssm 不可用且降级也失败
        """
        super().__init__()
        ...

    def forward(self, relational_features: Tensor) -> Tensor:
        """
        Args:
            relational_features: (T, GNN_HIDDEN_DIM)

        Returns:
            compressed_state: (SSM_STATE_DIM,)  跨帧时序压缩表征

        Raises:
            RuntimeError: T=0 时调用

        Tested-by: test_encoder.py::test_ssm_forward
        """
        ...


class FallbackSSMLayer(nn.Module):
    """纯 PyTorch 降级实现（不依赖 mamba-ssm）"""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        ...
```

---

## da_ocl/core/rule_constraints.py

```python
class RuleConstraintLayer(nn.Module):
    def __init__(self, num_rules: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.num_rules = num_rules

    def forward(self, ssm_output: Tensor) -> Tensor:
        """
        Args:
            ssm_output: (SSM_STATE_DIM,) SSM 压缩后的表征

        Returns:
            rule_out: (RULE_DIM,) 元规则约束后的慢记忆表征

        Raises:
            RuntimeError: ssm_output dim 不匹配

        Tested-by: test_encoder.py::test_rules_forward
        """
        ...
```

---

## da_ocl/core/composite_builder.py

```python
class CompositeFeatureBuilder:

    @staticmethod
    def build(o_batch: Tensor, o_hat_batch: Tensor) -> Tensor:
        """
        构建因果窗口级别的复合特征。

        B = concat(O_batch, O_hat_batch, O_batch - O_hat_batch)

        Args:
            o_batch: (SSM_STATE_DIM,) 当前窗口实际慢记忆表征
            o_hat_batch: (SSM_STATE_DIM,) 检索到的规律表征

        Returns:
            composite: (COMPOSITE_DIM,) 其中 COMPOSITE_DIM = 1536

        Raises:
            AssertionError: shape 不符合契约

        Tested-by: test_day_phase.py::test_composite_shape
        """
        residual = o_batch - o_hat_batch
        composite = torch.cat([o_batch, o_hat_batch, residual], dim=-1)
        assert composite.shape[-1] == COMPOSITE_DIM
        return composite
```

---

## da_ocl/memory/fast_memory.py

```python
class FastMemory(nn.Module):
    """
    基于现代霍普菲尔德网络（MHN）的快记忆。
    """

    def __init__(self, capacity: int = 5000) -> None:
        super().__init__()
        self.capacity = capacity
        # 存储槽
        self.register_buffer("keys", torch.zeros(capacity, COMPOSITE_DIM))
        self.register_buffer("values", torch.zeros(capacity, COMPOSITE_DIM))
        self.register_buffer("usage", torch.zeros(capacity))  # 使用计数

    def store(self, composite_features: List[Tensor]) -> None:
        """
        存储一个因果窗口的联合表征。

        Args:
            composite_features: List[(COMPOSITE_DIM,)] 长度为 T 的列表

        Raises:
            RuntimeError: 超出容量时（应触发替换）
        """
        ...

    def retrieve(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        """
        联想检索。

        Args:
            query: (COMPOSITE_DIM,) 查询向量

        Returns:
            (best_match, attention_weights)

        Raises:
            RuntimeError: 未存储任何内容时调用

        Tested-by: test_memory.py::test_fast_memory_retrieve
        """
        ...

    @property
    def stored_count(self) -> int:
        """当前存储的窗口数量"""
        ...
```

---

## da_ocl/memory/slow_memory.py

```python
class SlowMemory(nn.Module):
    """
    慢记忆三层结构容器。
    """

    def __init__(self, use_mamba: bool = True) -> None:
        super().__init__()
        self.gnn = GNNRelationEncoder()
        self.ssm = SSMCompressor(use_mamba)
        self.rules = RuleConstraintLayer()

    def forward(self, entity_window: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        三层级联前向。

        Args:
            entity_window: (T, entity_dim)

        Returns:
            gnn_out: (T, GNN_HIDDEN_DIM)
            ssm_out: (SSM_STATE_DIM,)
            rule_out: (RULE_DIM,)

        Tested-by: test_memory.py::test_slow_memory_forward
        """
        ...

    def retrieve(self, entity_window: Tensor) -> Tensor:
        """
        慢记忆检索路径（白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天白天）。

        Args:
            entity_window: (T, entity_dim)

        Returns:
            o_hat: (SSM_STATE_DIM,) 规律激活向量

        Note: 简化实现：直接返回 SSM 压缩输出作为规律检索结果。
              完整实现可在此处增加规律库索引和相似度检索。
        """
        ...

    def silent_write(self, entity_window: Tensor, mask_value: float) -> None:
        """
        沉默写入（白天联合掩码控制）。

        Args:
            entity_window: (T, entity_dim)
            mask_value: 联合掩码标量 ∈ [0,1]
        """
        ...

    def get_state_dict(self) -> SlowMemoryState: ...
    def load_state_dict(self, state_dict: SlowMemoryState) -> None: ...
```

---

## da_ocl/memory/joint_mask.py

```python
def compute_joint_mask(
    surprise_per_frame: Tensor,
    coverage_rate: float,
) -> float:
    """
    联合掩码计算。

    mask_batch = max_t(sigmoid(Surp_t)) × (1 - CoverageRate_batch)

    Args:
        surprise_per_frame: (T,) 每帧惊喜度
        coverage_rate: float ∈ [0,1]

    Returns:
        mask_scalar: float ∈ [0,1]

    Raises:
        ValueError: coverage_rate 不在 [0,1] 范围内

    Tested-by: test_memory.py::test_joint_mask_bounds
    """
    peak = torch.sigmoid(surprise_per_frame).max()
    mask = peak * (1.0 - coverage_rate)
    if not (0.0 <= mask <= 1.0):
        raise ValueError(f"mask_value {mask} out of bounds [0,1]")
    return mask.item()
```

---

## da_ocl/causal/window_manager.py

```python
class CausalWindowManager:

    def __init__(self, window_size: int = WINDOW_SIZE_DEFAULT) -> None:
        self.window_size = window_size
        self._buffer: List[Tensor] = []  # FIFO 队列

    def push(self, entity: Tensor) -> EntityWindow:
        """
        推送一帧到窗口。

        Args:
            entity: (entity_dim,) 单帧编码后实体

        Returns:
            EntityWindow: 当前完整窗口（固定长度 T）

        Raises:
            AssertionError: entity dim 不匹配

        Tested-by: test_causal.py::test_window_fifo
        """
        ...

    def get_batch(self) -> Tensor | None:
        """
        返回当前窗口张量。

        Returns:
            (T, entity_dim) 或 None（窗口未满时返回 None）
        """
        ...

    def reset(self) -> None: ...
```

---

## da_ocl/causal/boundary_detector.py

```python
class BoundaryDetector:

    def __init__(
        self,
        n_sigma: float = BOUNDARY_N_SIGMA,
        history_size: int = BOUNDARY_HISTORY_SIZE,
    ) -> None:
        ...

    def detect(self, batch_surprise: float) -> bool:
        """
        检测是否触发情景边界。

        条件: batch_surprise > mu_history + n_sigma * sigma_history

        Args:
            batch_surprise: float 窗口级别的惊喜度

        Returns:
            True: 触发新情景边界
            False: 同一情景继续

        Raises:
            RuntimeError: history 长度不足 10 时（预热期不触发）

        Tested-by: test_causal.py::test_boundary_no_false_positive
        """
        ...
```

---

## da_ocl/causal/surprise_detector.py

```python
class SurpriseDetector:

    @staticmethod
    def compute_window_surprise(o_batch: Tensor, o_hat_batch: Tensor) -> Tuple[float, Tensor]:
        """
        Surp_batch = ||O_batch - O_hat_batch||_2

        Args:
            o_batch: (SSM_STATE_DIM,)
            o_hat_batch: (SSM_STATE_DIM,)

        Returns:
            (scalar, per_frame)

        Raises:
            AssertionError: shape 不匹配
        """
        diff = o_batch - o_hat_batch
        scalar = torch.norm(diff, p=2).item()
        T = o_batch.shape[-1] // 3  # 反推 T（简化实现）
        per_frame = torch.full((T,), scalar / T)
        return scalar, per_frame
```

---

## da_ocl/causal/coverage_tracker.py

```python
class CoverageTracker:
    """
    追踪慢记忆对各因果规律的覆盖程度。
    用于计算 CoverageRate_batch（联合掩码的第二个因子）。
    """

    def __init__(self) -> None:
        self._coverage_map: Dict[str, float] = {}  # rule_id → coverage ∈ [0,1]
        self._global_mean: float = 0.0

    def update(self, activated_rules: List[str]) -> None:
        """
        根据当前窗口激活的规则更新覆盖图。
        """
        ...

    def get_batch_coverage_rate(self) -> float:
        """
        返回当前窗口的规律覆盖缺口。
        即：1 - global_mean_coverage
        """
        return max(0.0, 1.0 - self._global_mean)

    def get_global_mean(self) -> float:
        """返回全局平均覆盖率"""
        return self._global_mean
```

---

## da_ocl/buffer/three_zone_buffer.py

```python
@dataclass
class Experience:
    """Buffer 中存储的最小单位"""
    window: Tensor                  # (T, entity_dim)
    composite: Tensor               # (COMPOSITE_DIM,)
    surprise: float
    importance: float
    coverage_rate: float
    r_penalty: float
    is_death: bool = False
    causal_chain: List[int] | None = None
    timestep: int = 0
    priority_score: float = 0.0
    pattern_overlap: float = 0.0
    task_value: float = 0.0

    def to_dict(self) -> Dict: ...
    @classmethod
    def from_dict(cls, d: Dict) -> Experience: ...


class ThreeZoneBuffer:

    def __init__(self, total_capacity: int = 10000) -> None:
        ...

    def add(self, experience: Experience, zone_hint: str | None = None) -> bool:
        """
        尝试添加一条经验。

        Returns:
            True: 成功写入
            False: 经验被过滤（overlap 过高等原因）

        Raises:
            ValueError: zone_hint 不合法
        """
        ...

    def sample(
        self,
        num: int,
        priority: Literal["protected_first", "diversity", "task"] = "protected_first",
    ) -> List[Experience]:
        """
        采样经验用于夜晚蒸馏。

        Args:
            num: 采样数量
            priority: 采样优先级策略

        Returns:
            经验列表（可能少于 num，如果 Buffer 不足）

        Raises:
            RuntimeError: Buffer 完全为空时
        """
        ...

    def total_window_count(self) -> int:
        """返回 Buffer 中存储的窗口总数"""
        ...

    def get_state(self) -> BufferState: ...
    def load_state(self, state: BufferState) -> None: ...
    def rebalance(self, coverage_global_mean: float) -> None:
        """
        根据规律库成熟度动态调整各区容量。

        规律库越空 → 多样性区占比越大
        规律库越满 → 任务导向区占比越大
        """
        ...
```

---

## da_ocl/buffer/double_buffer.py

```python
class DoubleBufferManager:

    def __init__(self, slow_memory: SlowMemory) -> None:
        self._buffer_A = copy.deepcopy(slow_memory.state_dict())
        self._buffer_B: SlowMemoryState | None = None
        self._is_night_active = False

    def get_active_state(self) -> SlowMemoryState:
        """
        获取当前服务白天的稳定版本。
        夜晚激活期间持续返回 buffer_A。
        """
        return self._buffer_A

    def start_night_update(self) -> SlowMemoryState:
        """
        开始夜晚更新：复制 A 到 B。
        之后白天检索继续使用 A，夜晚更新操作 B。
        """
        if self._is_night_active:
            raise RuntimeError("夜晚更新已在进行中")
        self._buffer_B = copy.deepcopy(self._buffer_A)
        self._is_night_active = True
        return self._buffer_B

    def commit(self, new_slow_memory: SlowMemory) -> None:
        """
        夜晚整合完成且验证通过：原子替换 A ← B。
        更新慢记忆对象到最新参数。
        """
        assert self._is_night_active and self._buffer_B is not None
        self._buffer_A = self._buffer_B
        self._buffer_B = None
        self._is_night_active = False
        # 将 A 的状态加载到 slow_memory 对象
        slow_memory.load_state_dict(self._buffer_A)

    def rollback(self) -> None:
        """
        夜晚整合失败：丢弃 B，保留 A。
        掩码历史保留，下一次整合继续累积。
        """
        self._buffer_B = None
        self._is_night_active = False

    @property
    def is_night_active(self) -> bool:
        return self._is_night_active
```

---

## da_ocl/day/day_pipeline.py

```python
class DayPipeline:

    def __init__(
        self,
        encoder: Encoder,
        window_manager: CausalWindowManager,
        slow_memory: SlowMemory,
        fast_memory: FastMemory,
        joint_mask_fn: Callable,
        composite_builder: CompositeFeatureBuilder,
        adversarial_filter: AdversarialFilter,
        metrics_calculator: MetricsCalculator,
        boundary_detector: BoundaryDetector,
        coverage_tracker: CoverageTracker,
        buffer: ThreeZoneBuffer,
    ) -> None:
        ...

    def step(
        self,
        obs: Tensor,
        action: int,
        reward: float,
        done: bool,
    ) -> DayMetrics:
        """
        白天单步流程（完整并行+串行编排）。

        Returns:
            DayMetrics 字典

        Raises:
            ValueError: obs shape 不正确
        """
        ...
```

---

## da_ocl/night/night_pipeline.py

```python
class NightPipeline:

    def __init__(
        self,
        slow_memory: SlowMemory,
        fast_memory: FastMemory,
        buffer: ThreeZoneBuffer,
        loss_functions: NightLossFunctions,
        gradient_projector: GradientProjector,
        double_buffer: DoubleBufferManager,
        consistency_gate: ConsistencyGate,
    ) -> None:
        ...

    def step(self) -> NightMetrics:
        """
        夜晚单次整合流程。

        Returns:
            NightMetrics

        Raises:
            RuntimeError: 掩码历史为空
        """
        ...
```

---

## 异常处理约定

| 异常类型 | 何时抛出 | 处理方式 |
|----------|----------|----------|
| `ValueError` | 参数值超出有效范围 | 业务层捕获并记录日志，返回默认值或跳过 |
| `RuntimeError` | 系统状态不一致（如未初始化调用） | 传播给上层，训练流程打印警告并跳过该步 |
| `AssertionError` | 维度契约检查失败（开发阶段） | 转为 `ValueError` 后再抛出 |
| `FileNotFoundError` | checkpoint 加载失败 | 传播给上层 |
| `ImportError` | 可选依赖缺失 | 降级到备用实现，不影响主流程 |

---

*SPEC 版本：v0.1 | 待各 Milestone 逐步填充接口细节*
