# Old Prob 完整实现文档

## 概述

Old Prob（旧策略概率）是ROLL框架中用于计算off-policy ratio的关键组件。本文档详细说明了old prob的完整实现，包括两个维度的配置选项。

关于off-policy ratio的完整数据流和监控指标，请参阅[off_policy_ratio_dataflow_analysis.md](./off_policy_ratio_dataflow_analysis.md)。

## 配置选项

### 1. old_prob_mode（计算范围）

控制计算old log prob时覆盖的范围：

- **`trajectory`**（默认）：计算整个轨迹中所有assistant回复的log prob
- **`turn`**：仅计算最后一个assistant回复（当前轮次）的log prob

### 2. old_prob_compute（计算位置）

控制在哪里计算old log prob：

- **`trainer`**（默认）：在Actor-Train侧重新计算，更准确
- **`engine`**：使用推理引擎返回的log probs，更高效但可能不够准确

## 实现架构

### 1. Trainer模式实现

#### 1.1 数据流

```
生成阶段 → EnvManager → ReplayBuffer → 训练阶段
                           ↓
                    compute_log_probs
                    (Actor-Train)
```

#### 1.2 核心代码路径

**agentic_pipeline.py**：
```python
# 存储到replay buffer时计算behavior log probs
if old_prob_compute == "trainer":
    behavior_refs = self.actor_train.compute_log_probs(fresh_batch, blocking=False)
    behavior = DataProto.materialize_concat(data_refs=behavior_refs)
    fresh_batch.batch["behavior_log_probs"] = behavior.batch["log_probs"]
```

**base_worker.py**：
```python
def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
    # 检查是否需要使用turn模式
    old_prob_mode = data.meta_info.get("old_prob_mode", "trajectory")
    response_mask = data.batch["response_mask"]
    
    if old_prob_mode in ["step", "turn"] and "prompt_mask" in data.batch:
        # Turn模式：只计算最后一个assistant turn的log probs
        from roll.utils.turn_mode_utils import create_turn_mode_response_mask
        turn_response_mask, debug_info = create_turn_mode_response_mask(
            response_mask=response_mask,
            prompt_mask=data.batch.get("prompt_mask", None),
            messages_list=data.non_tensor_batch.get("messages_list", None)
        )
        response_mask_for_log_probs = turn_response_mask
    else:
        # Trajectory模式：使用原始response mask
        response_mask_for_log_probs = response_mask
```

### 2. Engine模式实现

#### 2.1 数据流

```
生成阶段 → VLLM/HF Engine → postprocess_generate → ReplayBuffer
            ↓                      ↓
         logprobs=1        generation_log_probs
```

#### 2.2 核心代码路径

**vllm_strategy.py**：
```python
def create_sampling_params_for_vllm(gen_kwargs):
    # 当选择engine计算路径时启用engine logprobs
    want_engine_logprobs = gen_kwargs.get("old_prob_compute", "trainer") == "engine"
    logprobs_flag = 1 if want_engine_logprobs else 0
    
    return SamplingParams(
        # ... 其他参数
        logprobs=logprobs_flag,
    )
```

**generate_scheduler.py**（推测实现）：
```python
# 在postprocess_output_ids中附加engine logprobs
if "output_logprobs" in response_data.meta_info:
    engine_logprobs = response_data.meta_info["output_logprobs"]
    # 处理并附加到output.batch["generation_log_probs"]
```

**agentic_pipeline.py**：
```python
# 使用engine提供的log probs
if old_prob_compute == "engine" and "generation_log_probs" in fresh_batch.batch:
    if old_prob_mode == "turn" and "prompt_mask" in fresh_batch.batch:
        # Apply turn mask to engine log probs
        turn_mask, _ = create_turn_mode_response_mask(
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            messages_list=messages_list
        )
        # Apply mask to engine log probs
        masked_log_probs = engine_log_probs * turn_mask.float()
        fresh_batch.batch["behavior_log_probs"] = masked_log_probs
    else:
        # Trajectory mode: use engine log probs directly
        fresh_batch.batch["behavior_log_probs"] = fresh_batch.batch["generation_log_probs"]
```

## Turn模式详解

### 1. 动机

在多轮对话场景中，每个动作都是基于当时的context生成的，而不是基于完整轨迹。Turn模式旨在更准确地反映这种情况。

### 2. 实现原理

**turn_mode_utils.py**：
```python
def create_turn_mode_response_mask(response_mask, prompt_mask, messages_list):
    """
    创建一个只标记最后一个assistant turn的response mask。
    
    原理：
    1. 扫描原始response_mask，找到所有assistant回复段
    2. 定位最后一个连续的1序列（最后一个assistant turn）
    3. 创建新mask，只在最后一个turn的位置标记为1
    """
```

### 3. 适用场景

- **TrajEnvManager**：需要动态创建turn mask
- **StepEnvManager**：天然支持，因为每个样本只包含当前轮次

## 配置示例

### 1. Trajectory + Trainer（默认）
```yaml
old_prob_mode: trajectory
old_prob_compute: trainer
```
最准确但计算开销较大。

### 2. Turn + Trainer
```yaml
old_prob_mode: turn
old_prob_compute: trainer
```
更准确地反映off-policy情况，适合多轮对话场景。

### 3. Trajectory + Engine
```yaml
old_prob_mode: trajectory
old_prob_compute: engine
```
使用引擎返回的log probs，效率高但可能不够准确。

### 4. Turn + Engine
```yaml
old_prob_mode: turn
old_prob_compute: engine
```
结合了turn模式的准确性和engine模式的效率。在存储到replay buffer时会应用turn mask，只保留最后一轮的log probs。

## 注意事项

1. **Engine模式的限制**：
   - 需要推理引擎支持返回log probs
   - VLLM通过设置`logprobs=1`参数启用
   - HF策略可能需要额外实现

2. **Turn模式的要求**：
   - 需要`prompt_mask`来识别轮次边界
   - TrajEnvManager和StepEnvManager都生成了必要的masks

3. **性能考虑**：
   - Trainer模式需要额外的前向传播
   - Engine模式直接使用生成时的log probs，更高效
   - Turn模式减少了需要计算的token数量

4. **准确性权衡**：
   - Trainer模式使用相同的模型权重重新计算，最准确
   - Engine模式可能因为量化、批处理等因素导致细微差异

## 当前实现状态

### 已实现功能

1. **Trainer模式**：
   - ✅ Trajectory模式：完整实现
   - ✅ Turn模式：完整实现（通过`turn_mode_utils.py`）

2. **Engine模式**：
   - ✅ VLLM策略：已实现提取log probs的代码
   - ✅ generate_scheduler：将engine返回的log probs附加到`generation_log_probs`
   - ✅ Turn模式兼容性：已支持，在存储到replay buffer时应用turn mask
   - ⚠️  需要验证：VLLM实际输出格式可能需要调整

3. **环境管理器支持**：
   - ✅ TrajEnvManager：生成`prompt_mask`，支持turn模式
   - ✅ StepEnvManager：天然支持turn模式，每个样本只包含当前轮次

### 待完善功能

1. **HF策略的Engine模式**：
   - 需要确认HF策略是否支持返回log probs
   - 可能需要额外实现

## Off-Policy监控指标

### 1. 指标概述

Off-policy监控系统通过比较behavior policy（行为策略，生成数据时的策略）和current policy（当前策略，训练时的策略）的log probabilities来评估replay训练的off-policy程度。

**核心实现**：`roll/pipeline/agentic/offpolicy_monitor.py`

### 2. 指标分类

#### 2.1 基础统计指标

**Log Ratio（对数比率）**：
- `{prefix}/log_ratio/mean` - 平均对数比率：`log(π_current/π_behavior)`
- `{prefix}/log_ratio/std` - 对数比率标准差
- `{prefix}/log_ratio/max` - 最大对数比率
- `{prefix}/log_ratio/min` - 最小对数比率

**Importance Ratio（重要性采样比率）**：
- `{prefix}/ratio/mean` - 平均比率：`exp(log_ratio)`
- `{prefix}/ratio/std` - 比率标准差
- `{prefix}/ratio/max` - 最大比率
- `{prefix}/ratio/min` - 最小比率
- `{prefix}/ratio/median` - 中位数比率
- `{prefix}/ratio/p95` - 95分位数
- `{prefix}/ratio/p05` - 5分位数
- `{prefix}/ratio/p99` - 99分位数

**解读**：
- `ratio = 1.0` 表示策略完全一致（on-policy）
- `ratio > 1.0` 表示当前策略更倾向于生成该action
- `ratio < 1.0` 表示当前策略不太倾向于生成该action
- 训练初期ratio接近1是正常的，随着训练进行会逐渐偏离

#### 2.2 Clipping分析

**Clip Fraction（裁剪比例）**：
- `{prefix}/ratio/clip_frac` - 被裁剪的token比例
- `{prefix}/ratio/clip_threshold` - 裁剪阈值（通常为0.2）
- `{prefix}/ratio/extreme_low_frac` - 极低比率（<0.5）的token比例
- `{prefix}/ratio/extreme_high_frac` - 极高比率（>2.0）的token比例

**解读**：
- `clip_frac` 高表示off-policy程度严重，可能需要：
  - 减小replay buffer容量
  - 增加训练频率（减小train_steps_per_env_step）
  - 使用importance sampling weights
- `extreme_*_frac` 帮助识别异常的off-policy样本

#### 2.3 有效性指标

**Effective Sample Size (ESS)**：
- `{prefix}/ess` - 有效样本大小：`(Σw)² / Σw²`
- `{prefix}/ess_ratio` - 有效样本比率：`ess / total_samples`

**解读**：
- ESS衡量重要性采样的有效性
- `ess_ratio = 1.0` 表示所有样本权重相等（完美on-policy）
- `ess_ratio` 越低，说明少数样本主导训练，可能需要：
  - 增加replay更新频率
  - 减小buffer capacity
  - 应用importance sampling truncation

**KL Divergence（KL散度）**：
- `{prefix}/kl_divergence` - 近似KL散度：`mean(ratio * log_ratio - (ratio - 1))`

**解读**：
- 衡量当前策略和行为策略的分布差异
- 值越大，off-policy程度越严重

#### 2.4 Token统计

**Token Counts（Token计数）**：
- `{prefix}/valid_tokens` - 有效token数量（response_mask=1的位置）
- `{prefix}/total_tokens` - 总token数量（包括prompt）
- `{prefix}/mask_rate` - 有效token占比：`valid_tokens / total_tokens`

**解读**：
- `mask_rate` 低表示大部分token是prompt，实际训练的response少
- 可以用来验证response_mask的正确性

### 3. 指标前缀

系统使用不同前缀区分数据来源：

- `fresh/offpolicy/*` - 新采样数据的off-policy指标（echo模式）
- `replay/offpolicy/*` - Replay buffer采样数据的off-policy指标

**为什么fresh也有offpolicy指标？**
- 在echo模式下，fresh数据先存入buffer再立即采样训练
- 由于存储和采样之间可能有微小的策略更新，也会有轻微的off-policy
- 通常`fresh/offpolicy/ratio/mean`应该非常接近1.0

### 4. 使用示例

#### 4.1 监控训练健康度

```python
# 在wandb或tensorboard中观察以下指标组合：
- replay/offpolicy/ratio/mean  # 应该在[0.8, 1.5]范围内
- replay/offpolicy/ratio/p95   # 不应该超过2.0
- replay/offpolicy/clip_frac   # 应该<0.1
- replay/offpolicy/ess_ratio   # 应该>0.5
- replay/offpolicy/kl_divergence  # 应该<0.5
```

#### 4.2 诊断Off-Policy问题

**症状1：ratio/mean远离1.0**
```
replay/offpolicy/ratio/mean: 1.8
replay/offpolicy/clip_frac: 0.35
```
**原因**：Replay buffer数据过旧，策略已大幅偏离
**解决方案**：
- 减小buffer capacity
- 增加replay更新频率
- 减少train_steps_per_env_step

**症状2：ess_ratio过低**
```
replay/offpolicy/ess_ratio: 0.2
replay/offpolicy/ratio/std: 0.8
```
**原因**：样本权重分布不均，少数样本主导训练
**解决方案**：
- 应用importance sampling weight clipping
- 使用更aggressive的sample_method（如LIFO）

**症状3：extreme_high_frac过高**
```
replay/offpolicy/ratio/extreme_high_frac: 0.15
```
**原因**：存在大量异常高权重样本
**解决方案**：
- 检查log_probs计算是否正确
- 应用ratio clipping (如PPO的clip_range)

#### 4.3 对比不同配置

```yaml
# 配置A：Trajectory + Trainer
old_prob_mode: trajectory
old_prob_compute: trainer

# 配置B：Turn + Engine
old_prob_mode: turn
old_prob_compute: engine
```

**预期差异**：
- Turn模式的`valid_tokens`会少于Trajectory模式（只计算最后一轮）
- Engine模式可能因为数值差异导致略高的`kl_divergence`
- 两种配置的`ratio/mean`应该接近（相差<0.05）

### 5. 实现细节

**代码位置**：`roll/pipeline/agentic/offpolicy_monitor.py`

**核心逻辑**：
```python
# 1. 提取log probs
behavior_log_probs = batch["old_log_probs"]  # 存储时的
current_log_probs = actor_train.compute_log_probs(batch)  # 当前策略重新计算

# 2. 应用response_mask
valid_behavior = behavior_log_probs[response_mask]
valid_current = current_log_probs[response_mask]

# 3. 计算ratio
log_ratio = valid_current - valid_behavior
ratio = torch.exp(log_ratio)

# 4. 统计指标
metrics = {
    "ratio/mean": ratio.mean().item(),
    "ratio/std": ratio.std().item(),
    "ess": compute_ess(ratio),
    # ... 更多指标
}
```

**调用时机**：
1. **主训练路径**（第250-260行）：计算`fresh/offpolicy/*`指标
2. **Replay训练循环**（第450-458行）：计算`replay/offpolicy/*`指标

### 6. 常见问题

**Q1: 为什么训练初期所有ratio都是1.0？**
- **A**: 正常现象。训练初期策略几乎没变化，所以current和behavior policy相同。等待几十个step后会逐渐偏离。

**Q2: fresh/offpolicy和replay/offpolicy有什么区别？**
- **A**:
  - `fresh` - 刚采样的数据（echo模式下也会经过buffer）
  - `replay` - 从buffer中采样的历史数据
  - `fresh`的off-policy程度应该远小于`replay`

**Q3: 为什么wandb只显示前25步的数据？**
- **A**: Wandb配置为offline模式，需要手动sync：
  ```bash
  wandb sync /path/to/wandb/offline-run-xxx
  ```

**Q4: valid_tokens为什么这么少？**
- **A**: 检查：
  - `mask_rate` 是否合理（应该>0.01）
  - `old_prob_mode` 是否为turn（会减少valid tokens）
  - response是否太短

## 未来改进方向

1. **Engine模式的完整实现**：
   - 确保HF策略也支持返回generation_log_probs
   - 处理不同推理引擎的log probs格式差异
   - 修复Engine + Turn模式的兼容性问题

2. **Turn模式的优化**：
   - 缓存turn mask的计算结果
   - 支持更复杂的turn识别逻辑

3. **监控和验证**：
   - 添加metrics来比较不同模式的off-policy ratio
   - 验证engine和trainer模式计算结果的一致性

4. **文档和测试**：
   - 添加单元测试覆盖所有模式组合
   - 提供性能基准测试结果

## 相关文档

- [off_policy_ratio_dataflow_analysis.md](./off_policy_ratio_dataflow_analysis.md) - Off-Policy Ratio完整数据流分析和监控指标体系
- [replay_buffer.md](./replay_buffer.md) - Replay Buffer实现详解
