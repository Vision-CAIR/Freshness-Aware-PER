# Replay Buffer 与 Off-Policy Monitor 配置指南

本文档详细介绍 ROLL 框架中 Replay Buffer 和 Off-Policy Monitor 的所有配置参数，包括参数含义、推荐值、以及参数之间的关系。

---

## 目录

1. [核心概念](#核心概念)
2. [Off-Policy Monitor 配置](#off-policy-monitor-配置)
3. [Replay Buffer 基础配置](#replay-buffer-基础配置)
4. [Age-Based Freshness 配置](#age-based-freshness-配置)
5. [Priority 优先采样配置](#priority-优先采样配置)
6. [Off-Policy Filter 配置](#off-policy-filter-配置)
7. [参数关系与调优建议](#参数关系与调优建议)
8. [配置示例](#配置示例)

---

## 核心概念

### Off-Policy 训练

Off-Policy 训练指使用**行为策略（behavior policy）**收集的数据来训练**当前策略（current policy）**。当两个策略差异较大时，需要通过 **Importance Sampling** 进行校正。

**关键公式**：
```
Importance Ratio = π_current(a|s) / π_behavior(a|s)
```

在 ROLL 中使用 **几何平均**（seq mode）计算 sample-level ratio：
```python
# 正确实现（几何平均）
masked_log_ratio = (log_ratio * response_mask).sum(dim=1) / valid_tokens
per_sample_ratio = torch.exp(masked_log_ratio)
```

### Replay Buffer 核心指标

基于论文 "Revisiting Fundamentals of Experience Replay" (Fedus et al., ICML 2020)：

| 指标 | 公式 | 含义 |
|------|------|------|
| **Replay Capacity** | `capacity` | Buffer 能存储的样本数量 |
| **Replay Ratio** | `train_steps_per_env_step` | 每次环境交互后的训练次数 |
| **Oldest Policy Age** | `capacity / (batch_size × replay_ratio)` | Buffer 中最旧数据的"年龄" |

**关键结论**：数据新鲜度（Oldest Policy Age）是影响 off-policy 训练的隐藏关键因素。

---

## Off-Policy Monitor 配置

```yaml
offpolicy_monitor:
  enabled: true
  save_behavior_log_probs: true
  monitor_fresh_batch: true
  monitor_replay_batch: true
  monitor_interval: 1
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 off-policy 监控 |
| `save_behavior_log_probs` | bool | `true` | 是否保存行为策略的 log_probs 到 replay buffer |
| `monitor_fresh_batch` | bool | `true` | 是否监控新鲜数据（刚收集的） |
| `monitor_replay_batch` | bool | `true` | 是否监控 replay buffer 采样的数据 |
| `monitor_interval` | int | `1` | 监控频率（每 N 步监控一次） |

### 监控指标说明

Monitor 会输出以下指标（所有指标以 `offpolicy/` 为前缀）：

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `offpolicy/ratio_mean` | 平均 importance ratio | 0.8 ~ 1.2 |
| `offpolicy/ratio_std` | ratio 标准差 | < 0.5 |
| `offpolicy/ratio_max` | 最大 ratio | < 3.0 |
| `offpolicy/ratio_min` | 最小 ratio | > 0.3 |
| `offpolicy/kl_divergence` | 当前策略与行为策略的 KL 散度 | < 0.1 |
| `offpolicy/clipped_fraction` | 被 clip 的 token 比例 | < 0.2 |

**异常信号**：
- `ratio_mean` 持续偏离 1.0 → 策略变化过快
- `ratio_max` 频繁 > 5.0 → 数据过于陈旧
- `kl_divergence` > 0.5 → off-policy 程度过高

---

## Replay Buffer 基础配置

```yaml
replay:
  enabled: true
  capacity: 10000
  min_size: ${rollout_batch_size}
  train_steps_per_env_step: 2
  use_rollout_batch_size: true
  sampling_mode: "trajectory"
  storage_mode: tokens_only
  lazy_tokenization: false
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用 replay buffer |
| `capacity` | int | `10000` | Buffer 容量（样本数） |
| `min_size` | int | `batch_size` | 开始采样的最小 buffer 大小 |
| `train_steps_per_env_step` | int | `1` | 每次环境交互后从 buffer 训练的次数 |
| `use_rollout_batch_size` | bool | `true` | 采样时是否使用 rollout batch size |
| `sampling_mode` | str | `"trajectory"` | 采样模式：`trajectory`（轨迹级）或 `step`（步级） |
| `storage_mode` | str | `tokens_only` | 存储模式 |
| `lazy_tokenization` | bool | `false` | 是否延迟 tokenization |

### capacity 参数分析

**计算实际存储能力**：
```
实际存储 batches = capacity / rollout_batch_size
例如：10000 / 128 = 78 个 batch
```

**推荐设置**：
- 小规模实验：`capacity = 5000 ~ 10000`
- 大规模训练：`capacity = 50000 ~ 100000`

**注意**：capacity 过大会导致数据过于陈旧（off-policy 程度高），需要配合 `age_decay` 调整。

### train_steps_per_env_step 参数分析

**训练流程**：
```
每个 global step:
1. 收集新数据 → 训练 1 次（fresh batch）
2. 从 buffer 采样 → 训练 N 次（replay batch）
总训练次数 = 1 + train_steps_per_env_step
```

**推荐设置**：
- 保守：`1`（1 fresh + 1 replay = 2 次）
- 标准：`2`（1 fresh + 2 replay = 3 次）
- 激进：`4`（1 fresh + 4 replay = 5 次，需要更激进的 age_decay）

**权衡**：
- 值越大 → 样本利用率越高 → 但 off-policy 程度也越高
- 需要配合 `age_decay` 确保数据新鲜度

---

## Age-Based Freshness 配置

```yaml
replay:
  age_decay: 40.0
```

### age_decay 参数详解

**衰减公式**：
```
effective_priority = intrinsic_priority × exp(-age / age_decay)
```

其中 `age = current_global_step - stored_at_step`

**衰减效果对照表**：

| age_decay | age=20 | age=40 | age=80 | age=120 |
|-----------|--------|--------|--------|---------|
| **40** | 0.61 | **0.37** | 0.14 | 0.05 |
| **100** | 0.82 | 0.67 | 0.45 | 0.30 |
| **500** | 0.96 | 0.92 | 0.85 | 0.79 |
| **1500** | 0.99 | 0.97 | 0.95 | 0.92 |

**理论依据**（来自 "Revisiting Fundamentals of Experience Replay"）：
- 数据新鲜度是 off-policy 训练的关键因素
- `age_decay` 应该与你希望数据保持"有效"的时间尺度匹配
- 当 `age = age_decay` 时，权重降为 `1/e ≈ 0.37`

**推荐设置**：
```
age_decay ≈ capacity / (batch_size × 2)
```

例如：`10000 / (128 × 2) = 39 ≈ 40`

| 场景 | capacity | batch_size | 推荐 age_decay |
|------|----------|------------|----------------|
| 小 buffer | 5000 | 128 | 20 ~ 30 |
| 中 buffer | 10000 | 128 | 30 ~ 50 |
| 大 buffer | 50000 | 128 | 100 ~ 200 |

---

## Priority 优先采样配置

```yaml
replay:
  use_advantage_priority: true

  priority:
    function: reward
    alpha: 1.0
    use_importance_weights: true
    importance_beta: 0.4
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_advantage_priority` | bool | `false` | 是否在训练后使用 advantage 更新优先级 |
| `priority.function` | str | `"uniform"` | 初始优先级函数 |
| `priority.alpha` | float | `1.0` | 优先级指数（PER 中的 α） |
| `priority.use_importance_weights` | bool | `true` | 是否使用重要性采样权重 |
| `priority.importance_beta` | float | `0.4` | IS 权重指数（PER 中的 β） |

### priority.function 可选值

| 函数名 | 说明 | 适用场景 |
|--------|------|----------|
| `uniform` | 均匀采样（所有样本优先级相同） | 基础实验 |
| `lifo` | 后进先出（最新数据优先） | Echo 模式，近 on-policy |
| `fifo` | 先进先出（最旧数据优先） | 确保所有数据被使用 |
| `reward` | 基于 \|reward\| 的优先级 | 关注高奖励样本 |
| `recency` | 基于年龄的指数衰减 | 关注新鲜数据 |
| `combined` | reward + recency 组合 | 平衡方案 |
| `advantage` | 基于 \|advantage\| | 需要训练后更新 |
| `td_error` | 基于 TD error（标准 PER） | 需要 value function |

### priority.alpha 参数

控制优先级的"锐度"：
- `alpha = 0`：完全均匀采样（忽略优先级）
- `alpha = 0.5`：中等优先级效果
- `alpha = 1.0`：完全按优先级采样

**采样概率公式**：
```
P(i) = priority_i^alpha / Σ(priority_j^alpha)
```

### priority.importance_beta 参数

控制 IS 权重的强度，用于校正优先采样带来的偏差：
- `beta = 0`：不使用 IS 权重（有偏）
- `beta = 1.0`：完全校正（无偏但方差大）
- `beta = 0.4 ~ 0.6`：推荐范围

**IS 权重公式**：
```
w_i = (1 / (N × P(i)))^beta
```

**退火策略**：
- 训练初期：`beta = 0.4`（允许一定偏差，降低方差）
- 训练后期：`beta → 1.0`（完全无偏）

### use_advantage_priority 两阶段策略

启用后的优先级更新流程：

```
阶段1（Push 时）：
  priority = |episode_reward| + epsilon  # 使用 reward 作为初始优先级

阶段2（训练后）：
  priority = |computed_advantage| + epsilon  # 使用计算出的 advantage 更新
```

**优势**：
- 初始时没有 advantage，使用 reward 作为代理
- 训练后有了更准确的 advantage 估计，更新优先级
- 类似 PER 中的 TD-error 更新机制

---

## Off-Policy Filter 配置

```yaml
replay:
  enable_offpolicy_filter: true
  ratio_clip_max: 3.0
  filter_mini_batch_size: 32
  filter_max_attempts: 20
  filter_min_acceptable_batch: 64

  # 自适应配置
  filter_adaptive_mini_batch: true
  filter_success_rate_window: 5
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_offpolicy_filter` | bool | `false` | 是否启用 off-policy 过滤 |
| `ratio_clip_max` | float | `3.0` | 最大允许的 importance ratio |
| `filter_mini_batch_size` | int | `32` | 过滤时的 mini-batch 大小 |
| `filter_max_attempts` | int | `20` | 最大尝试采样次数 |
| `filter_min_acceptable_batch` | int | `64` | 最小可接受的有效样本数 |
| `filter_adaptive_mini_batch` | bool | `true` | 是否启用自适应 mini-batch |
| `filter_success_rate_window` | int | `5` | 成功率滑动窗口大小 |

### ratio_clip_max 参数

**过滤条件**：
```python
# 样本被过滤掉如果：
importance_ratio > ratio_clip_max  # ratio 太大，数据太旧
# 或
importance_ratio < 1/ratio_clip_max  # ratio 太小，也是异常
```

**推荐设置**：
- 保守：`2.0`（严格过滤，保留更"新鲜"的数据）
- 标准：`3.0`（平衡过滤）
- 宽松：`5.0`（允许更旧的数据）

### 过滤流程

```
1. 从 buffer 采样 mini_batch_size 个样本
2. 计算每个样本的 importance ratio（几何平均）
3. 过滤掉 ratio > ratio_clip_max 的样本
4. 重复直到收集够 target_batch_size 个有效样本
5. 如果达到 max_attempts 仍不够：
   - 有效样本 >= min_acceptable_batch → 使用现有样本
   - 有效样本 < min_acceptable_batch → 补充未过滤样本
   - 有效样本 = 0 → 回退到无过滤采样
```

### 自适应 Mini-Batch 策略

根据历史成功率动态调整 `filter_mini_batch_size`：

| 平均成功率 | 调整策略 | 说明 |
|------------|----------|------|
| ≥ 95% | `mini_batch = target_batch_size` | 一次采样完成 |
| ≥ 90% | `mini_batch *= 2` | 翻倍增长 |
| ≥ 70% | `mini_batch *= 1.5` | 适度增长 |
| < 70% | `mini_batch = initial_size` | 回退到初始值 |

**优势**：
- 高成功率时减少采样次数，提高效率
- 低成功率时保持小 batch，避免浪费 GPU 计算

---

## 参数关系与调优建议

### 参数关系图

```
                    ┌─────────────────┐
                    │    capacity     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   age_decay     │ │ train_steps_per │ │  batch_size     │
│                 │ │   _env_step     │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Oldest Policy   │
                    │     Age         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Off-Policy      │
                    │   Degree        │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ratio_clip_max  │ │ importance_beta │ │  Filter         │
│                 │ │                 │ │  Success Rate   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 调优建议

#### 场景1：训练不稳定，loss 震荡

**症状**：
- `offpolicy/ratio_max` 频繁 > 5.0
- `offpolicy/kl_divergence` > 0.3
- 训练 loss 大幅波动

**调整**：
1. 减小 `capacity`（减少旧数据）
2. 减小 `age_decay`（加快旧数据降权）
3. 减小 `ratio_clip_max`（更严格过滤）
4. 减小 `train_steps_per_env_step`（减少 off-policy 训练）

#### 场景2：样本利用率低，收敛慢

**症状**：
- 训练进度慢
- GPU 利用率低
- 每个样本只被使用 1-2 次

**调整**：
1. 增大 `train_steps_per_env_step`（更多重复训练）
2. 增大 `capacity`（存储更多样本）
3. 适当增大 `age_decay`（保留更多数据）

#### 场景3：Filter 成功率低

**症状**：
- 日志显示 filter 成功率 < 50%
- 频繁 fallback 到无过滤采样

**调整**：
1. 增大 `ratio_clip_max`（放宽过滤条件）
2. 减小 `age_decay`（让 buffer 中数据更新鲜）
3. 减小 `capacity`（减少旧数据积累）

### 推荐配置组合

#### 保守配置（稳定性优先）

```yaml
replay:
  capacity: 5000
  train_steps_per_env_step: 1
  age_decay: 25.0
  enable_offpolicy_filter: true
  ratio_clip_max: 2.0
  priority:
    function: uniform
    alpha: 0.5
```

#### 标准配置（平衡方案）

```yaml
replay:
  capacity: 10000
  train_steps_per_env_step: 2
  age_decay: 40.0
  enable_offpolicy_filter: true
  ratio_clip_max: 3.0
  use_advantage_priority: true
  priority:
    function: reward
    alpha: 1.0
    importance_beta: 0.4
```

#### 激进配置（样本效率优先）

```yaml
replay:
  capacity: 20000
  train_steps_per_env_step: 4
  age_decay: 50.0
  enable_offpolicy_filter: true
  ratio_clip_max: 5.0
  use_advantage_priority: true
  priority:
    function: reward
    alpha: 1.0
    importance_beta: 0.6
```

---

## 配置示例

### 完整配置示例

```yaml
# =============================================================================
# OFF-POLICY MONITORING
# =============================================================================
offpolicy_monitor:
  enabled: true                    # 启用监控
  save_behavior_log_probs: true    # 保存行为策略 log_probs
  monitor_fresh_batch: true        # 监控新鲜数据
  monitor_replay_batch: true       # 监控 replay 数据
  monitor_interval: 1              # 每步监控

# =============================================================================
# TRAJECTORY REPLAY BUFFER
# =============================================================================
replay:
  # --- 基础配置 ---
  enabled: true
  capacity: 10000                  # Buffer 容量（~78 batches @128）
  min_size: ${rollout_batch_size}  # 开始采样的最小大小
  train_steps_per_env_step: 2      # Replay ratio = 2.0
  use_rollout_batch_size: true
  sampling_mode: "trajectory"      # 轨迹级采样
  storage_mode: tokens_only
  lazy_tokenization: false

  # --- Age-Based Freshness ---
  # 公式: effective_priority = intrinsic_priority × exp(-age / age_decay)
  # age_decay=40 时，40 步后权重降为 0.37，80 步后降为 0.14
  age_decay: 40.0

  # --- Priority 配置 ---
  use_advantage_priority: true     # 训练后用 advantage 更新优先级
  priority:
    function: reward               # 初始用 |reward| 作为优先级
    alpha: 1.0                     # 完全按优先级采样
    use_importance_weights: true   # 使用 IS 权重校正
    importance_beta: 0.4           # IS 权重指数

  # --- Off-Policy Filter ---
  enable_offpolicy_filter: true
  ratio_clip_max: 3.0              # 最大允许 ratio
  filter_mini_batch_size: 32       # Mini-batch 大小
  filter_max_attempts: 20          # 最大尝试次数
  filter_min_acceptable_batch: 64  # 最小可接受样本数

  # --- 自适应 Mini-Batch ---
  filter_adaptive_mini_batch: true
  filter_success_rate_window: 5    # 成功率窗口大小
```

---

## 参考文献

1. Fedus, W., et al. (2020). "Revisiting Fundamentals of Experience Replay". ICML 2020.
   - 核心发现：Replay Capacity、Replay Ratio、Oldest Policy Age 三者相互关联
   - 关键结论：数据新鲜度是隐藏的关键因素

2. Schaul, T., et al. (2016). "Prioritized Experience Replay". ICLR 2016.
   - PER 原始论文
   - 提出 α（优先级指数）和 β（IS 权重指数）

3. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms".
   - PPO 算法
   - Importance Sampling 在 policy gradient 中的应用
