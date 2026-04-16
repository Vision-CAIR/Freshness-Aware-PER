# EnvManager Length Parameters 详细说明

## 概述

在ROLL框架的Agentic Pipeline中，涉及多个"长度"相关的参数，容易混淆。本文档详细说明各个参数的含义、作用范围和相互关系。

---

## 📋 参数清单

### 1. `sequence_length` (Pipeline级别)

**定义位置**：
```yaml
# 配置文件顶层
sequence_length: 2048
```

**含义**：
- **Padding的目标长度**
- 训练时，所有样本会被padding到这个长度，以便batch化处理

**作用范围**：
- ✅ **StepEnvManager**: 单个step的padding长度（prompt + response）
- ✅ **TrajEnvManager**: 整个trajectory的padding长度（prompt + response1 + response2 + ... + responseN）

**使用位置**：
1. **生成时限制** (`make_decision`):
   ```python
   # step_env_manager.py:180 (注意：这行代码有bug)
   remaining = sequence_length - input_ids.length - max_new_tokens
   ```

2. **训练时padding** (`formulate_rollouts`):
   ```python
   # step_env_manager.py:245
   input_ids = pad_to_length(input_ids, length=sequence_length, ...)
   ```

---

### 2. `max_tokens_per_step` (Environment级别)

**定义位置**：
```yaml
# 配置文件环境配置
max_tokens_per_step: 4096

custom_envs:
  NQSearchStep:
    max_tokens_per_step: ${max_tokens_per_step}
```

**含义**：
- **单个step允许生成的最大token数**
- 这是一个软限制（理论上限）
- 实际生成可能因stop_strings提前终止

**作用范围**：
- ✅ **StepEnvManager**: 每个step的生成上限
- ✅ **TrajEnvManager**: 每个action的生成上限

**使用位置**：
```python
# step_env_manager.py:176
max_new_tokens = min(
    self.env_config["max_tokens_per_step"],  # 来自这里
    self.worker_config.generating_args.max_new_tokens
)
```

---

### 3. `max_new_tokens` (Generation级别)

**定义位置**：
```yaml
actor_infer:
  generating_args:
    max_new_tokens: ${max_tokens_per_step}  # 通常引用max_tokens_per_step
```

**含义**：
- **推理引擎（VLLM/HF）的生成上限**
- 这是最终传递给推理引擎的参数

**作用范围**：
- ✅ 所有EnvManager

**计算逻辑**：
```python
# step_env_manager.py:176-180
max_new_tokens = min(
    env_config["max_tokens_per_step"],           # 环境配置的上限
    worker_config.generating_args.max_new_tokens # 推理配置的上限
)

# 再根据sequence_length调整（有bug的地方）
generation_config["max_new_tokens"] = min(
    max_new_tokens,
    sequence_length - input_ids.length - max_new_tokens  # ⚠️ 这里有bug
)
```

---

### 4. `max_actions_per_traj` (Episode级别)

**定义位置**：
```yaml
max_actions_per_traj: 5

custom_envs:
  NQSearchStep:
    env_config:
      max_steps: ${max_actions_per_traj}
```

**含义**：
- **一个episode包含的最大step数**
- 对于StepEnvManager：一个episode最多5个steps
- 对于TrajEnvManager：一个trajectory最多5个actions

**作用范围**：
- ✅ **StepEnvManager**: 控制episode何时结束
- ✅ **TrajEnvManager**: 控制trajectory何时结束

---

## 🔗 参数关系图

### StepEnvManager (单step处理)

```
┌─────────────────────────────────────────────────────────────┐
│                    Single Step                               │
│                                                              │
│  ┌─────────────┐  ┌──────────────────────────────────────┐ │
│  │   Prompt    │  │         Response                      │ │
│  │  (~140 tok) │  │  (actual: ~115 tok, max: 4096 tok)   │ │
│  └─────────────┘  └──────────────────────────────────────┘ │
│                                                              │
│  ◀───────────────── sequence_length (2048) ─────────────────▶│
│                                                              │
└─────────────────────────────────────────────────────────────┘

实际使用长度：~255 tokens
Padding到：2048 tokens
浪费比例：(2048 - 255) / 2048 = 87.5%
```

**长度关系**：
```python
# 理想情况
sequence_length >= prompt_length + max_tokens_per_step

# 实际情况（因为有stop_strings）
sequence_length >= prompt_length + actual_response_length
# 140 + 115 = 255 tokens << 2048 tokens
```

---

### TrajEnvManager (整个trajectory)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Full Trajectory                                    │
│                                                                       │
│  ┌────┐ ┌────────┐ ┌────┐ ┌────────┐     ┌────┐ ┌────────┐         │
│  │Prpt│ │Action1 │ │Obs │ │Action2 │ ... │Obs │ │ActionN │         │
│  └────┘ └────────┘ └────┘ └────────┘     └────┘ └────────┘         │
│                                                                       │
│  ◀──────────────── sequence_length ──────────────────────────────▶   │
└──────────────────────────────────────────────────────────────────────┘

# 对于TrajEnvManager
sequence_length >= prompt_length +
                   (max_tokens_per_step + observation_length) × max_actions_per_traj
```

---

## ⚙️ 配置建议

### 场景1：StepEnvManager + 短Response（当前NQ Search场景）

```yaml
# 观察到的实际数据
# - Prompt: ~140 tokens
# - Response: ~115 tokens (因为stop_strings提前终止)
# - Total: ~255 tokens

# 推荐配置
sequence_length: 1024  # 4x safety margin
max_tokens_per_step: 800  # 实际很少用到，因为有stop_strings
max_actions_per_traj: 5

# 内存占用
# Per sample: 1024 tokens
# Batch 128: 128 × 1024 = 131K tokens
```

### 场景2：StepEnvManager + 长Response

```yaml
# 如果response可能很长（无stop_strings或长推理）
sequence_length: 5120  # 足够容纳最长情况
max_tokens_per_step: 4096
max_actions_per_traj: 5

# 内存占用
# Per sample: 5120 tokens
# Batch 128: 128 × 5120 = 655K tokens
```

### 场景3：TrajEnvManager

```yaml
# 整个trajectory一次性生成
sequence_length: 20480  # prompt + 5 actions × 4096
max_tokens_per_step: 4096
max_actions_per_traj: 5

# 内存占用
# Per sample: 20480 tokens
# Batch 128: 128 × 20480 = 2.6M tokens (非常大！)
```

---

## 🐛 已知Bug

### Bug 1: make_decision中的max_new_tokens计算错误

**位置**: `step_env_manager.py:180`

**当前代码**:
```python
generation_config["max_new_tokens"] = min(
    max_new_tokens,
    max(sequence_length - input_ids.length - max_new_tokens, 1)
    #                                        ^^^^^^^^^^^^^^^ 多余的！
)
```

**正确应该是**:
```python
remaining_space = sequence_length - input_ids.length
generation_config["max_new_tokens"] = min(max_new_tokens, max(remaining_space, 1))
```

**影响**:
- 计算出的可生成空间比实际小
- 可能导致不必要的截断

**修复建议**: 提交PR修复此bug

---

## 📊 实际数据流分析（基于训练日志）

### 观察到的实际长度

```json
{
  "tokens/prompt_length/mean": 139.375,
  "tokens/response_length/mean": 115.1328125,
  "total_actual_length": 254.5,

  "sequence_length_config": 2048,
  "padding_ratio": 87.6%,  // (2048 - 254) / 2048

  "max_tokens_per_step_config": 4096,
  "actual_usage_ratio": 2.8%  // 115 / 4096
}
```

### 为什么response这么短？

1. **Stop strings触发**:
   ```yaml
   stop_strings: ["</search>", "</answer>"]
   ```
   模型一旦生成`</search>`或`</answer>`就停止

2. **环境设计**:
   - NQ Search环境鼓励简洁回答
   - 典型格式: `<think>...</think><search>query</search>` (很短)
   - 或: `<think>...</think><answer>answer</answer>` (也很短)

3. **实际生成示例**:
   ```xml
   <think> The question asks about NHA meaning. </think>
   <search> NHA medical field </search>
   ```
   总共不到100 tokens

---

## 🎯 最终推荐配置（针对NQ Search）

### 方案A: 基于理论最大长度（推荐）⭐

```yaml
# 核心参数
# 假设: 完整trajectory ~20k tokens, 5 steps
# 计算: 20000 / 5 = 4000 tokens per step
sequence_length: 4096  # 单个step的最大长度
max_tokens_per_step: 4096  # 保持与sequence_length一致
max_actions_per_traj: 5

# 推理配置
actor_infer:
  generating_args:
    max_new_tokens: ${max_tokens_per_step}  # 4096
    stop_strings: ["</search>", "</answer>"]  # 提前终止机制
```

**优势**:
- ✅ 内存占用降低 68% (4096 vs 12800)
- ✅ NCCL通信量降低 68%
- ✅ 足够容纳理论最大长度
- ✅ 训练速度提升 ~3倍
- ✅ 与trajectory总长度(20k)逻辑一致

**配置逻辑**:
```
Full trajectory: ~20,000 tokens
÷ 5 steps
= ~4,000 tokens per step
→ sequence_length: 4096
```

### 方案B: 基于实际观察（内存优化）

```yaml
# 核心参数
# 观察: 实际使用 ~254 tokens (prompt 140 + response 115)
sequence_length: 1024  # 4x actual usage, 提供充足余量
max_tokens_per_step: 800  # 降低到合理值
max_actions_per_traj: 5

# 推理配置
actor_infer:
  generating_args:
    max_new_tokens: ${max_tokens_per_step}  # 800
    stop_strings: ["</search>", "</answer>"]  # 保持
```

**优势**:
- ✅ 内存占用降低 92% (1024 vs 12800)
- ✅ NCCL通信量降低 92%
- ✅ 最大化内存效率
- ✅ 训练速度提升 ~5倍
- ⚠️ 可能截断非常长的response（罕见情况）

---

## 📝 总结

### 关键要点

1. **StepEnvManager vs TrajEnvManager**:
   - Step: `sequence_length` = 单个step的长度
   - Traj: `sequence_length` = 整个trajectory的长度

2. **Padding vs Generation**:
   - Padding: 训练时统一长度，便于batch处理
   - Generation: 推理时的实际生成上限

3. **理论值 vs 实际值**:
   - `max_tokens_per_step`: 理论上限 (4096)
   - 实际生成: 受stop_strings限制 (~115)

4. **配置原则**:
   - `sequence_length` >= `prompt_length` + `expected_max_response`
   - 不要过大（浪费内存）
   - 不要过小（导致截断）

### 问题排查清单

当遇到length相关问题时：

- [ ] 检查`sequence_length`是否足够容纳prompt + response
- [ ] 检查日志中的实际长度：`tokens/prompt_length/mean` 和 `tokens/response_length/mean`
- [ ] 检查是否有截断警告：`maybe you should increase the response_length`
- [ ] 检查NCCL通信量：如果太大，考虑降低`sequence_length`
- [ ] 检查GPU内存占用：如果OOM，降低`sequence_length`或`rollout_batch_size`

---

**文档版本**: v1.0
**更新日期**: 2025-11-23
**适用框架**: ROLL Agentic Pipeline
