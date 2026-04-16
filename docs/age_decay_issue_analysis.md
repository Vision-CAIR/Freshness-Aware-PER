# Age Decay 实现问题分析报告

## 0. 训练流程时序分析

### 0.1 单步训练时序图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Training Step Timeline                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Time ──────────────────────────────────────────────────────────────────────►   │
│                                                                                  │
│  GPU:  ┌──────────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────┐ │
│        │   Rollout    │     │  Ref/Old  │     │   Actor     │     │  Replay  │ │
│        │   (VLLM)     │     │  LogProbs │     │   Train     │     │  Train   │ │
│        └──────────────┘     └───────────┘     └─────────────┘     └──────────┘ │
│                                                                                  │
│  CPU:  ┌──────────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────┐ │
│        │   Env Mgr    │     │ Advantage │     │   IDLE ⭐   │     │  Sample  │ │
│        │   (Async)    │     │   Compute │     │             │     │  Buffer  │ │
│        └──────────────┘     └───────────┘     └─────────────┘     └──────────┘ │
│                                                                                  │
│        ▲                                       ▲                                 │
│        │                                       │                                 │
│        └─ CPU partially busy                   └─ CPU IDLE: Best time for       │
│           (env interactions)                      age decay refresh!            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 0.2 可并行刷新的时间窗口

| 阶段 | GPU 状态 | CPU 状态 | 可并行刷新? |
|------|----------|----------|-------------|
| Rollout (VLLM) | Busy | Partially Busy | 🟡 可能 |
| Ref/Old LogProbs | Busy | Computing Adv | 🟡 可能 |
| **Actor Train** | **Busy** | **IDLE** | ✅ **最佳** |
| Store to Buffer | Idle | Busy | ❌ 不行 |
| Replay Train | Busy | Sample/Prepare | 🟡 采样前 |

### 0.3 代码位置参考

```python
# agentic_pipeline.py 主训练循环
def run(self):
    for global_step in range(max_steps):
        # Phase 1: Rollout (Line 251-254)
        batch = ray.get(self.train_rollout_scheduler.get_batch.remote(...))

        # Phase 2: Ref/Old LogProbs (Line 290-356)
        ref_log_probs_refs = self.reference.compute_log_probs(batch, blocking=False)
        old_log_probs_refs = self.actor_train.compute_log_probs(batch, blocking=False)

        # Phase 3: Advantage (Line 358-591) - CPU
        batch = compute_advantage(...)

        # Phase 4: Actor Train (Line 621) - GPU busy, CPU IDLE ⭐
        actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
        # ← 此时 CPU 空闲，可以刷新 age decay！
        actor_train_metrics = DataProto.materialize_concat(actor_train_metrics_refs)

        # Phase 5: Store (Line 694)
        self.store_fresh_data_to_replay_buffer(batch, global_step)

        # Phase 6: Replay Train (Line 716-onwards)
        for step_idx in range(train_steps_per_env_step):
            mb = self.replay_buffer.sample_for_training(...)  # ← 采样前刷新！
            self.actor_train.train_step(mb)
```

---

## 1. 问题背景

### 1.1 Age Decay 的设计目标

在 Off-policy RL 中，样本的"新鲜度"（freshness）对训练效果有重要影响。越老的样本与当前策略的偏差越大（policy drift），可能导致训练不稳定。Age decay 的目标是：

- **自动降低老样本的采样优先级**
- **优先使用新鲜样本**，减少 off-policy 偏差
- **结合 reward-based priority**，平衡样本重要性和新鲜度

### 1.2 预期行为

理想的 age decay 应该是**动态的**：

```
样本存入时: priority = reward
时间流逝后: effective_priority = reward * exp(-age / age_decay)
```

随着时间推移，老样本的有效优先级应该自动降低，无论是否被采样过。

---

## 2. 当前实现分析

### 2.1 代码位置

| 文件 | 功能 |
|------|------|
| `roll/agentic/replay_buffer/priority_functions.py` | Priority 计算函数 |
| `roll/agentic/replay_buffer/step_buffer.py` | Step-level Replay Buffer |
| `roll/agentic/replay_buffer/trajectory_buffer.py` | Trajectory-level Replay Buffer |

### 2.2 两个 Age Decay 相关的配置

```yaml
# 配置 1: priority_function
priority_function: reward_fresh  # 使用 reward_fresh_priority 函数

# 配置 2: enable_age_decay
enable_age_decay: true           # 在 update_priorities 时应用 age decay
age_decay: 500.0                 # 衰减常数
```

### 2.3 `reward_fresh_priority` 函数

```python
# priority_functions.py:275-336
def reward_fresh_priority(entry, global_step, epsilon=1e-6, age_decay=500.0, **kwargs):
    # 计算 reward
    reward_component = abs(reward) + epsilon

    # 计算 age decay
    age = max(0, global_step - entry.stored_at_step)
    freshness_weight = np.exp(-age / age_decay)

    # 组合
    priority = reward_component * freshness_weight
    return float(priority)
```

### 2.4 存储流程

```python
# step_buffer.py:193-243

# Step 1: 创建 entry，设置 stored_at_step = global_step
step_entry = StepEntry(
    ...
    stored_at_step=global_step,  # Line 216
    global_step=global_step      # Line 218
)

# Step 2: 调用 priority_fn 计算优先级
priority = self.priority_fn(step_entry, global_step, age_decay=self.age_decay)  # Line 224
step_entry.priority = float(priority)  # Line 225

# Step 3: Segment tree 使用 max_priority（标准 PER）
priority_alpha = self._max_priority ** self.priority_exponent  # Line 241
self._it_sum[current_idx] = priority_alpha  # Line 242 - 不是 step_entry.priority！
```

### 2.5 update_priorities 流程

```python
# step_buffer.py:854-876

for idx, priority in zip(indices, priorities):
    # 更新 entry.priority
    buffer_list[idx].priority = priority  # Line 863

    # 计算 effective priority（可选 age decay）
    if self.enable_age_decay:  # Line 866
        sample_age = global_step - buffer_list[idx].global_step
        freshness_weight = np.exp(-sample_age / self.age_decay)
        effective_priority = priority * freshness_weight
    else:
        effective_priority = priority

    # 更新 segment tree
    priority_alpha = effective_priority ** self.priority_exponent
    self._it_sum[idx] = priority_alpha  # Line 875
```

---

## 3. 问题详解

### 3.1 问题 1：`reward_fresh_priority` 在存储时无效

**原因**：存储时 `stored_at_step = global_step`，所以：

```python
age = global_step - entry.stored_at_step  # = 0
freshness_weight = np.exp(-0 / age_decay)  # = 1.0
priority = reward * 1.0                     # = reward
```

**结论**：存储时 `reward_fresh_priority` 等价于 `reward_priority`。

### 3.2 问题 2：priority_fn 输出被 max_priority 覆盖

**原因**：标准 PER 做法是新样本使用 `max_priority` 确保至少被采样一次：

```python
# step_buffer.py:241-243
priority_alpha = self._max_priority ** self.priority_exponent
self._it_sum[current_idx] = priority_alpha  # 不使用 step_entry.priority
```

**结论**：`priority_fn()` 的计算结果在存储时被忽略。

### 3.3 问题 3：Age decay 只对被采样的样本有效

**原因**：`enable_age_decay` 只在 `update_priorities()` 中生效，而 `update_priorities()` 只对被采样的样本调用。

```python
# 只更新被采样的 indices
for idx, priority in zip(indices, priorities):  # indices 是采样的样本
    if self.enable_age_decay:
        # 只有这些样本会应用 age decay
```

**结论**：未被采样的老样本，其 segment tree 值不会随时间更新。

### 3.4 问题 4：Segment tree 不会自动刷新

**场景说明**：

```
时间 T=0:   样本A存入, segment_tree[A] = max_priority^α = 1.0
时间 T=100: 样本B存入, segment_tree[B] = max_priority^α = 1.0
时间 T=200: 采样，只采到B
            调用 update_priorities([B], [0.5], global_step=200)

            样本B: age = 200-100 = 100
                   freshness = exp(-100/500) = 0.82
                   segment_tree[B] = (0.5 * 0.82)^0.6 = 0.52

            样本A: segment_tree[A] = 仍然是 1.0 ← 问题！
                   理论上 age = 200, freshness = 0.67
                   应该是 segment_tree[A] ≈ 0.67^0.6 = 0.78
```

**结论**：老样本的 segment tree 值没有因为时间流逝而降低，导致采样概率偏高。

---

## 4. 问题影响

### 4.1 对实验的影响

| 配置 | 预期效果 | 实际效果 |
|------|----------|----------|
| `reward_fresh` + `enable_age_decay: false` | reward × age_decay | 只有 reward |
| `reward_fresh` + `enable_age_decay: true` | reward × age_decay (动态) | 只对被采样的样本有 age_decay |
| `reward` + `enable_age_decay: true` | reward × age_decay (动态) | 只对被采样的样本有 age_decay |

### 4.2 01/24 step_reward_fresh 实验分析

```yaml
# 实际使用的配置
priority_function: reward_fresh
enable_age_decay: false
```

**实际效果**：
1. 存储时：`reward_fresh(age=0)` = `reward`
2. Segment tree：使用 `max_priority`
3. update_priorities：使用 advantage，**无 age decay**

**结论**：这个实验**完全没有使用 age decay**！

---

## 5. 解决方案

### 5.1 方案对比

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A: 周期性全量刷新** | 每 N 步刷新所有样本的 age decay | 实现简单，行为正确 | O(buffer_size)，可能影响性能 |
| **A+: 异步并行刷新** ⭐ | 在 GPU 训练时异步刷新 | 零额外延迟，行为正确 | 需要线程管理 |
| **B: 采样前局部刷新** | 采样前刷新候选样本的 age decay | 性能较好 | 实现复杂 |
| **C: 延迟计算** | 采样时动态计算 age decay | 最准确 | 需要修改 segment tree |
| **D: 保持现状 + 文档** | 接受当前行为，正确文档化 | 无代码改动 | 功能有限 |

### 5.2 推荐方案：A+ - 异步并行刷新 ⭐

**核心思想**：利用 GPU 训练期间 CPU 空闲的时间窗口，异步刷新 age decay。

```
Timeline:
┌───────────────────────────────────────────────────────────────┐
│  GPU:  │████ Actor Train ████│                                │
│  CPU:  │░░░ Age Refresh ░░░░░│  (parallel, no extra latency)  │
└───────────────────────────────────────────────────────────────┘
```

**优点**：
1. **零额外延迟**：在 GPU 训练时并行执行，不阻塞主流程
2. **行为正确**：所有样本的 age decay 都会被刷新
3. **实现简单**：使用 Python ThreadPoolExecutor 或 Ray

### 5.3 方案 A+ 实现思路

#### 5.3.1 Replay Buffer 层：添加刷新方法

```python
# 在 BaseReplayBuffer 中添加
def refresh_all_age_decay(self, current_global_step: int) -> None:
    """
    刷新所有样本的 age decay，更新 segment tree。

    时间复杂度: O(buffer_size)
    典型耗时: buffer_size=100000 时约 50-100ms
    """
    if not self.enable_age_decay:
        return

    buffer_list = list(self.steps) if hasattr(self, 'steps') else self.trajectories

    for idx, entry in enumerate(buffer_list):
        if entry is None or not self.is_valid(idx):
            continue

        # 计算当前 age
        age = current_global_step - entry.global_step
        freshness_weight = np.exp(-age / self.age_decay)

        # 计算 effective priority
        effective_priority = entry.priority * freshness_weight

        # 更新 segment tree
        priority_alpha = max(effective_priority, 1e-8) ** self.priority_exponent
        self._it_sum[idx] = priority_alpha
        self._it_min[idx] = priority_alpha

    logger.debug(f"Refreshed age decay for {len(buffer_list)} samples at step {current_global_step}")
```

#### 5.3.2 Pipeline 层：异步并行调用

```python
# 在 agentic_pipeline.py 中
from concurrent.futures import ThreadPoolExecutor

class AgenticPipeline(BasePipeline):
    def __init__(self, ...):
        ...
        # 创建线程池用于异步刷新
        self._age_decay_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="age_decay")
        self._age_decay_future = None

    def _async_refresh_age_decay(self, global_step: int) -> None:
        """异步刷新 age decay（在 GPU 训练时并行执行）"""
        if not self.pipeline_config.replay.enabled:
            return
        if not self.pipeline_config.replay.enable_age_decay:
            return

        # 提交异步任务
        self._age_decay_future = self._age_decay_executor.submit(
            self.replay_buffer.refresh_all_age_decay, global_step
        )

    def _wait_age_decay_refresh(self) -> None:
        """等待 age decay 刷新完成"""
        if self._age_decay_future is not None:
            self._age_decay_future.result()  # 等待完成
            self._age_decay_future = None
```

#### 5.3.3 集成到训练循环

```python
# 在 run() 方法中
def run(self):
    for global_step in range(max_steps):
        ...

        # Phase 4: Actor Train - GPU busy, CPU IDLE
        actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)

        # ⭐ 在 GPU 训练时启动异步 age decay 刷新
        if self.pipeline_config.replay.enabled and global_step % refresh_interval == 0:
            self._async_refresh_age_decay(global_step)

        # 等待训练完成
        actor_train_metrics = DataProto.materialize_concat(actor_train_metrics_refs)

        ...

        # Phase 6: Replay Train
        if self.pipeline_config.replay.enabled:
            # ⭐ 在采样前确保刷新完成
            self._wait_age_decay_refresh()

            for step_idx in range(train_steps_per_env_step):
                mb = self.replay_buffer.sample_for_training(...)  # 使用刷新后的优先级
                self.actor_train.train_step(mb)
```

#### 5.3.4 配置扩展

```yaml
replay:
  enabled: true
  priority_function: reward      # 不需要 reward_fresh
  enable_age_decay: true
  age_decay: 500.0
  refresh_interval: 1            # 每步刷新（推荐），或设为 5-10 减少开销
```

### 5.4 性能估算

#### 刷新耗时估算

| Buffer Size | 刷新耗时（估算） | GPU 训练耗时（参考） | 是否可并行隐藏 |
|-------------|------------------|---------------------|---------------|
| 10,000 | ~5-10ms | ~500ms | ✅ 完全隐藏 |
| 50,000 | ~25-50ms | ~500ms | ✅ 完全隐藏 |
| 100,000 | ~50-100ms | ~500ms | ✅ 大部分隐藏 |
| 500,000 | ~250-500ms | ~500ms | 🟡 可能有少量开销 |

**结论**：对于典型的 buffer size（10k-100k），刷新耗时可以完全被 GPU 训练时间隐藏。

#### 内存开销

- 无额外内存开销（原地更新 segment tree）
- 线程池开销：~1MB

### 5.5 替代方案：Ray 异步（可选）

如果需要更好的并发控制，可以使用 Ray：

```python
@ray.remote
def refresh_age_decay_task(replay_buffer, global_step):
    replay_buffer.refresh_all_age_decay(global_step)

# 在 pipeline 中
refresh_ref = refresh_age_decay_task.remote(self.replay_buffer, global_step)
# ... GPU 训练 ...
ray.get(refresh_ref)  # 等待完成
```

### 5.6 其他修改建议

1. **删除或标记 `reward_fresh_priority` 为 deprecated**
   - 这个函数在当前设计中没有实际作用
   - 避免用户混淆

2. **更新文档**
   - 明确说明 `enable_age_decay` 的工作机制
   - 提供正确的配置示例

3. **添加监控指标**
   - 记录 buffer 中样本的平均 age
   - 记录 age decay 刷新的频率和耗时

---

## 6. 正确的配置示例

### 6.1 启用 Age Decay（修复后）

```yaml
replay:
  enabled: true
  capacity: 100000
  priority_function: reward       # 使用 reward，不是 reward_fresh
  priority_exponent: 0.6
  enable_age_decay: true          # 启用动态 age decay
  age_decay: 500.0                # 半衰期 ≈ 346 步
  refresh_interval: 1             # 每步刷新（异步，无额外开销）
  importance_sampling_correction: true
  importance_beta: 0.4
```

### 6.2 不使用 Age Decay（标准 PER）

```yaml
replay:
  enabled: true
  capacity: 100000
  priority_function: reward
  priority_exponent: 0.6
  enable_age_decay: false         # 标准 PER，无 age decay
  importance_sampling_correction: true
  importance_beta: 0.4
```

### 6.3 废弃的配置（不推荐）

```yaml
# ❌ 不推荐：reward_fresh 在当前实现中无效
replay:
  priority_function: reward_fresh  # 存储时 age=0，等于 reward
  enable_age_decay: false          # 无动态刷新
```

---

## 7. 总结

### 7.1 核心问题

1. **`reward_fresh_priority` 函数在存储时无效**（age=0）
2. **Segment tree 不会自动刷新**，老样本优先级不会随时间降低
3. **Age decay 只对被采样的样本有效**

### 7.2 根本原因

当前实现将 age decay 的计算放在了**更新时**（update_priorities），而不是**采样时**。这导致未被采样的样本无法享受 age decay 的降权效果。

### 7.3 推荐修复

添加**周期性全量刷新机制**，在采样前定期更新所有样本的 age decay，确保老样本的采样优先级能够正确降低。

---

## 8. 实现清单

### 8.1 需要修改的文件

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `roll/agentic/replay_buffer/step_buffer.py` | 添加 `refresh_all_age_decay()` 方法 | P0 |
| `roll/agentic/replay_buffer/trajectory_buffer.py` | 添加 `refresh_all_age_decay()` 方法 | P0 |
| `roll/agentic/replay_buffer/base_buffer.py` | 添加抽象方法定义（可选） | P1 |
| `roll/pipeline/agentic/agentic_pipeline.py` | 添加异步刷新逻辑 | P0 |
| `roll/pipeline/agentic/agentic_config.py` | 添加 `refresh_interval` 配置 | P1 |

### 8.2 实现步骤

```
Step 1: Replay Buffer 层
├── step_buffer.py
│   └── 添加 refresh_all_age_decay(self, current_global_step: int) 方法
│       ├── 遍历所有 steps
│       ├── 计算 age = current_global_step - entry.global_step
│       ├── 计算 freshness = exp(-age / age_decay)
│       ├── 更新 segment tree: _it_sum[idx] = (priority * freshness)^α
│       └── 返回刷新的样本数量（用于监控）
│
└── trajectory_buffer.py
    └── 同上，遍历 trajectories

Step 2: Pipeline 层
├── agentic_pipeline.py
│   ├── __init__(): 创建 ThreadPoolExecutor
│   ├── _async_refresh_age_decay(): 提交异步任务
│   ├── _wait_age_decay_refresh(): 等待任务完成
│   └── run(): 在 actor_train.train_step() 后调用异步刷新
│
└── agentic_config.py
    └── ReplayConfig: 添加 refresh_interval 字段

Step 3: 测试验证
├── 单元测试：验证 refresh_all_age_decay() 正确更新 segment tree
├── 集成测试：验证异步刷新不影响训练流程
└── 性能测试：验证刷新耗时 < GPU 训练耗时
```

### 8.3 关键代码片段

#### Step Buffer 刷新方法

```python
# step_buffer.py - 在 StepReplayBuffer 类中添加

def refresh_all_age_decay(self, current_global_step: int) -> int:
    """
    刷新所有样本的 age decay，更新 segment tree。

    Args:
        current_global_step: 当前训练步数

    Returns:
        刷新的样本数量
    """
    if not self.enable_age_decay:
        return 0

    self.current_global_step = current_global_step
    buffer_list = list(self.steps)
    refreshed_count = 0

    for idx, entry in enumerate(buffer_list):
        if entry is None:
            continue

        # 计算当前 age 和 freshness
        age = max(0, current_global_step - entry.global_step)
        freshness_weight = np.exp(-age / self.age_decay)

        # 计算 effective priority
        effective_priority = max(entry.priority * freshness_weight, 1e-8)

        # 更新 segment tree
        priority_alpha = effective_priority ** self.priority_exponent
        self._it_sum[idx] = priority_alpha
        self._it_min[idx] = priority_alpha

        refreshed_count += 1

    logger.debug(
        f"Refreshed age decay for {refreshed_count} samples at step {current_global_step}, "
        f"age_decay={self.age_decay}"
    )

    return refreshed_count
```

#### Pipeline 异步刷新

```python
# agentic_pipeline.py - 在 AgenticPipeline 类中添加

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        ...

        # 创建线程池用于异步 age decay 刷新
        self._age_decay_executor: Optional[ThreadPoolExecutor] = None
        self._age_decay_future: Optional[Future] = None

        if self.pipeline_config.replay.enabled and self.pipeline_config.replay.enable_age_decay:
            self._age_decay_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="age_decay_refresh"
            )
            logger.info("Created ThreadPoolExecutor for async age decay refresh")

    def _async_refresh_age_decay(self, global_step: int) -> None:
        """在 GPU 训练时异步刷新 age decay"""
        if self._age_decay_executor is None:
            return
        if self.replay_buffer is None:
            return

        # 检查刷新间隔
        refresh_interval = getattr(self.pipeline_config.replay, 'refresh_interval', 1)
        if global_step % refresh_interval != 0:
            return

        # 等待上一次刷新完成（如果还在进行）
        self._wait_age_decay_refresh()

        # 提交新的刷新任务
        self._age_decay_future = self._age_decay_executor.submit(
            self.replay_buffer.refresh_all_age_decay,
            global_step
        )
        logger.debug(f"Started async age decay refresh at step {global_step}")

    def _wait_age_decay_refresh(self) -> None:
        """等待 age decay 刷新完成"""
        if self._age_decay_future is not None:
            try:
                result = self._age_decay_future.result(timeout=10.0)  # 10s 超时
                logger.debug(f"Age decay refresh completed, refreshed {result} samples")
            except Exception as e:
                logger.warning(f"Age decay refresh failed: {e}")
            finally:
                self._age_decay_future = None
```

#### 集成到训练循环

```python
# agentic_pipeline.py - run() 方法中修改

def run(self):
    for global_step in range(self.pipeline_config.max_steps):
        ...

        # Phase 4: Actor Train (Line ~621)
        if self.pipeline_config.critic_warmup <= global_step:
            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)

            # ⭐ 在 GPU 训练时启动异步 age decay 刷新
            if self.pipeline_config.replay.enabled:
                self._async_refresh_age_decay(global_step)

            actor_train_metrics = DataProto.materialize_concat(actor_train_metrics_refs)

        ...

        # Phase 6: Replay Train (Line ~697)
        if self.pipeline_config.replay.enabled:
            # ⭐ 在采样前确保刷新完成
            self._wait_age_decay_refresh()

            if self.replay_buffer.can_sample(batch_size=training_batch_size):
                for step_idx in range(rb_cfg.train_steps_per_env_step):
                    ...
```

### 8.4 配置字段

```python
# agentic_config.py - ReplayConfig 中添加

@dataclass
class ReplayConfig:
    ...

    refresh_interval: int = field(
        default=1,
        metadata={
            "help": "Interval (in steps) for refreshing age decay of all samples. "
                    "Set to 1 for every step (recommended with async refresh), "
                    "or higher values to reduce overhead."
        }
    )
```

### 8.5 监控指标

```python
# 在 run() 中添加监控
metrics.update({
    "replay_buffer/age_decay_refresh_count": refresh_count,
    "replay_buffer/age_decay_refresh_time_ms": refresh_time_ms,
    "replay_buffer/avg_sample_age": avg_age,
    "replay_buffer/max_sample_age": max_age,
})
```

---

## 附录：相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| reward_fresh_priority | priority_functions.py | 275-336 |
| 存储时计算 priority | step_buffer.py | 221-228 |
| 存储时设置 segment tree | step_buffer.py | 238-243 |
| update_priorities | step_buffer.py | 822-882 |
| 采样（PER） | step_buffer.py | 348-358 |
| 采样（Hierarchical） | step_buffer.py | 618-630 |
| Pipeline 训练循环 | agentic_pipeline.py | 224-1100 |
| Actor Train 调用 | agentic_pipeline.py | 621 |
| Replay Train 循环 | agentic_pipeline.py | 716-onwards |
