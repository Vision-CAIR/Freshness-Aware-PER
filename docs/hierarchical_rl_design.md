# Hierarchical RL Implementation Design

## 概述

本文档描述了ROLL框架中Hierarchical RL的完整设计方案。该方案实现了真正的两层强化学习架构：
- **上层（Step-level）**：处理环境奖励，评估状态价值
- **下层（Token-level）**：接收上层的intrinsic rewards，优化生成策略

**限制**：仅在使用`StepEnvManager`时可用。

## 核心思想

### 1. 传统实现的问题

当前ROLL的实现存在以下问题：
- N-step returns没有使用bootstrap（`use_bootstrap=false`）
- Token-level GAE独立计算，不考虑step-level的价值
- 上下两层没有有效连接，不是真正的hierarchical RL

### 2. 新方案的核心改进

```
环境交互（Step-level）
    ↓ 环境reward
计算Step-level Value/Advantage (GAE或N-step)
    ↓ Intrinsic Reward
Token生成（Token-level）
    ↓ 接收上层指导
计算Token-level Advantage
    ↓
策略更新
```

**关键**：Token-level的"reward"不再是简单的环境reward，而是来自Step-level的价值估计！

## 配置系统

### HierarchicalRLConfig

```yaml
hierarchical:
  # 主开关（仅StepEnvManager可用）
  enabled: true

  # ============================================
  # 上层（Step-level）配置
  # ============================================

  # 优势估计方法
  step_level_estimator: "gae"  # "gae" | "nstep" | "monte_carlo"

  # Step-level GAE参数
  step_gamma: 0.99      # step之间的discount
  step_lambda: 0.95     # GAE lambda

  # N-step returns参数（当estimator="nstep"时使用）
  step_n_steps: 5
  use_step_bootstrap: true

  # Step value提取方式
  step_value_source: "last_token"  # "last_token" | "mean_tokens" | "max_tokens"

  # ============================================
  # 下层（Token-level）配置
  # ============================================

  # 优势估计方法
  token_level_estimator: "gae"  # "gae" | "reinforce" | "reinforce_baseline" | "direct"

  # Token-level GAE参数
  token_gamma: 0.99     # token之间的discount
  token_lambda: 0.95    # GAE lambda

  # ============================================
  # Reward分配策略
  # ============================================

  # 如何将step-level的return分配到tokens
  reward_assignment: "last_token"  # "last_token" | "uniform" | "exponential" | "value_weighted"

  # value_weighted的温度参数
  assignment_temperature: 1.0

  # ============================================
  # 混合与消融实验
  # ============================================

  # 是否保留原始token rewards
  use_original_rewards: false

  # 混合权重（1.0=纯hierarchical, 0.0=纯原始）
  mixing_alpha: 1.0

  # ============================================
  # 日志
  # ============================================

  log_hierarchical_metrics: true
  debug_mode: false
```

## 核心组件设计

### 1. HierarchicalAdvantageComputer

主要计算类，负责两层advantage的计算。

```python
class HierarchicalAdvantageComputer:
    """
    两层Hierarchical RL的advantage计算器
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config
        self.step_computer = StepLevelComputer(config)
        self.token_computer = TokenLevelComputer(config)

    def compute(
        self,
        batch: Dict[str, torch.Tensor],
        critic_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        主入口：计算两层的advantages

        Args:
            batch: 包含rewards, masks等的批次数据
            critic_values: Critic模型输出的token-level values

        Returns:
            {
                'step_values': step-level values,
                'step_advantages': step-level advantages,
                'step_returns': step-level returns,
                'token_advantages': token-level advantages (最终用于训练),
                'token_returns': token-level returns,
                'metrics': 详细指标
            }
        """
        pass
```

### 2. StepLevelComputer

上层计算组件。

```python
class StepLevelComputer:
    """
    Step-level的advantage计算
    """

    def extract_step_values(
        self,
        token_values: torch.Tensor,
        response_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        从token values提取step values

        根据config.step_value_source:
        - last_token: 每个step最后一个token的value
        - mean_tokens: 所有token values的平均
        - max_tokens: 所有token values的最大值

        Returns:
            step_values: [batch_size] 每个step的value
        """
        pass

    def compute_step_advantages(
        self,
        env_rewards: torch.Tensor,
        step_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算step-level advantages和returns

        根据config.step_level_estimator:
        - "gae": 使用GAE算法
        - "nstep": 使用N-step returns with bootstrap
        - "monte_carlo": 纯Monte Carlo returns

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        pass

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step-level GAE实现

        δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        A_t = Σ (γλ)^k * δ_{t+k}
        """
        pass

    def compute_nstep_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        N-step returns with bootstrap

        R_t = Σ_{k=0}^{n-1} γ^k * r_{t+k} + γ^n * V(s_{t+n})
        """
        pass
```

### 3. TokenLevelComputer

下层计算组件。

```python
class TokenLevelComputer:
    """
    Token-level的advantage计算
    """

    def assign_rewards_to_tokens(
        self,
        step_returns: torch.Tensor,
        response_masks: torch.Tensor,
        token_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将step-level returns分配到tokens

        这是连接上下层的关键！

        根据config.reward_assignment:
        - "last_token": 只给最后一个token
        - "uniform": 均匀分配
        - "exponential": 指数衰减
        - "value_weighted": 根据value贡献加权

        Returns:
            token_intrinsic_rewards: [batch_size, seq_len]
        """
        pass

    def compute_token_advantages(
        self,
        intrinsic_rewards: torch.Tensor,
        token_values: torch.Tensor,
        response_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算token-level advantages

        根据config.token_level_estimator:
        - "gae": 使用GAE with intrinsic rewards
        - "reinforce": 纯Monte Carlo
        - "reinforce_baseline": REINFORCE with value baseline
        - "direct": 直接使用intrinsic rewards

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        pass

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token-level GAE实现

        在每个step内部独立计算
        """
        pass
```

## 实现细节

### 1. Step Value提取

**last_token**（推荐）：
```python
def extract_last_token_value(token_values, response_masks):
    """
    使用最后一个token的value作为step value

    理由：在autoregressive模型中，最后一个token能看到所有前文
    """
    batch_size = token_values.size(0)
    step_values = []

    for i in range(batch_size):
        # 找到最后一个有效token
        last_idx = response_masks[i].sum() - 1
        step_values.append(token_values[i, last_idx])

    return torch.stack(step_values)
```

**mean_tokens**：
```python
def extract_mean_value(token_values, response_masks):
    """
    所有token values的平均

    理由：减少单个token估计的方差
    """
    masked_values = token_values * response_masks
    step_values = masked_values.sum(dim=1) / response_masks.sum(dim=1).clamp(min=1)
    return step_values
```

### 2. Step-level GAE

```python
def compute_step_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    标准的GAE算法，但应用在step-level
    """
    advantages = []
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        if t < len(rewards) - 1:
            nextvalue = values[t + 1]
        else:
            nextvalue = 0  # Terminal

        # TD error
        delta = rewards[t] + gamma * nextvalue - values[t]

        # GAE accumulation
        lastgaelam = delta + gamma * lambda_ * lastgaelam
        advantages.append(lastgaelam)

    advantages = torch.stack(list(reversed(advantages)))
    returns = advantages + values

    return advantages, returns
```

### 3. Reward分配策略

**last_token**（最简单，推荐起始）：
```python
def assign_last_token(step_return, num_tokens):
    """
    只有最后一个token获得全部reward
    """
    token_rewards = torch.zeros(num_tokens)
    token_rewards[-1] = step_return
    return token_rewards
```

**uniform**（公平分配）：
```python
def assign_uniform(step_return, num_tokens):
    """
    均匀分配到所有tokens
    """
    return torch.ones(num_tokens) * (step_return / num_tokens)
```

**exponential**（更符合TD思想）：
```python
def assign_exponential(step_return, num_tokens, gamma=0.99):
    """
    从最后往前指数衰减

    最后token: step_return
    倒数第二: step_return * gamma
    ...
    """
    weights = torch.pow(gamma, torch.arange(num_tokens - 1, -1, -1))
    weights = weights / weights.sum()  # Normalize
    return step_return * weights
```

**value_weighted**（基于贡献）：
```python
def assign_value_weighted(step_return, token_values, temperature=1.0):
    """
    根据token values的贡献分配

    贡献大的token获得更多reward
    """
    # Softmax with temperature
    weights = torch.softmax(token_values / temperature, dim=0)
    return step_return * weights
```

### 4. Token-level Advantage计算

**GAE**（推荐，低方差）：
```python
def compute_token_gae(intrinsic_rewards, token_values, mask, gamma=0.99, lambda_=0.95):
    """
    使用intrinsic rewards计算token-level GAE
    """
    advantages = torch.zeros_like(intrinsic_rewards)

    for step_idx in range(batch_size):
        lastgaelam = 0
        step_advantages = []
        valid_len = int(mask[step_idx].sum())

        for t in reversed(range(valid_len)):
            if t < valid_len - 1:
                nextvalue = token_values[step_idx, t + 1]
            else:
                nextvalue = 0

            delta = intrinsic_rewards[step_idx, t] + gamma * nextvalue - token_values[step_idx, t]
            lastgaelam = delta + gamma * lambda_ * lastgaelam
            step_advantages.append(lastgaelam)

        # Fill in
        for i, adv in enumerate(reversed(step_advantages)):
            advantages[step_idx, i] = adv

    returns = advantages + token_values
    return advantages, returns
```

**REINFORCE with Baseline**（简单稳定）：
```python
def compute_reinforce_baseline(intrinsic_rewards, token_values, mask):
    """
    直接用intrinsic rewards减去value baseline
    """
    advantages = intrinsic_rewards - token_values
    advantages = advantages * mask  # Apply mask
    returns = intrinsic_rewards
    return advantages, returns
```

## Pipeline集成

### 1. 修改位置

主要修改`roll/pipeline/agentic/agentic_pipeline.py`:

```python
# 在train_step中
def train_step(self, batch):
    # 1. 获取critic values
    if self.pipeline_config.hierarchical.enabled:
        values_refs = self.critic.compute_values(batch, blocking=False)
        values = DataProto.materialize_concat(data_refs=values_refs)

        # 2. 计算hierarchical advantages
        hierarchical_computer = HierarchicalAdvantageComputer(
            self.pipeline_config.hierarchical
        )

        results = hierarchical_computer.compute(
            batch=batch.batch,
            critic_values=values.batch["values"]
        )

        # 3. 替换advantages
        batch.batch["advantages"] = results["token_advantages"]
        batch.batch["returns"] = results["token_returns"]

        # 4. 记录指标
        if self.pipeline_config.hierarchical.log_hierarchical_metrics:
            metrics.update(results["metrics"])

    else:
        # 原有的逻辑
        ...
```

### 2. StepReplayBuffer修改

在`roll/agentic/replay_buffer/step_buffer.py`的`sample`方法中：

```python
def sample(self, ...):
    # ... 原有采样逻辑 ...

    # 如果启用hierarchical，需要存储额外信息
    if self.hierarchical_enabled:
        # 存储每个step的环境reward（用于上层）
        dataproto.batch["env_rewards"] = torch.from_numpy(env_rewards)

        # 标记这是hierarchical模式
        dataproto.meta_info["hierarchical_mode"] = True

    return dataproto, sampled_indices
```

### 3. 配置验证

```python
def validate_config(pipeline_config):
    if pipeline_config.hierarchical.enabled:
        # 检查是否使用StepEnvManager
        if pipeline_config.env_manager_type != "step":
            raise ValueError(
                "Hierarchical RL only works with StepEnvManager, "
                f"got {pipeline_config.env_manager_type}"
            )

        # 检查是否启用replay buffer
        if not pipeline_config.replay.enabled:
            logger.warning(
                "Hierarchical RL recommended to use with replay buffer for better step-level learning"
            )
```

## 实验配置示例

### 配置1：标准Hierarchical GAE

```yaml
# experiments/nq_search_hierarchical/hierarchical_gae.yaml

# 环境管理器
env_manager_type: "step"

# Replay buffer
replay:
  enabled: true
  capacity: 100000
  train_steps_per_env_step: 2

# Hierarchical RL配置
hierarchical:
  enabled: true

  # 上层：GAE
  step_level_estimator: "gae"
  step_gamma: 0.99
  step_lambda: 0.95
  step_value_source: "last_token"

  # 下层：GAE
  token_level_estimator: "gae"
  token_gamma: 0.99
  token_lambda: 0.95

  # Reward分配
  reward_assignment: "last_token"

  # 日志
  log_hierarchical_metrics: true

# 原有的adv_estimator被hierarchical覆盖
# adv_estimator: "gae"  # 不再使用
```

### 配置2：上层N-step + 下层REINFORCE

```yaml
hierarchical:
  enabled: true

  # 上层：N-step returns
  step_level_estimator: "nstep"
  step_n_steps: 5
  use_step_bootstrap: true
  step_gamma: 0.99
  step_value_source: "last_token"

  # 下层：REINFORCE with baseline（更简单）
  token_level_estimator: "reinforce_baseline"

  # Reward分配：指数衰减
  reward_assignment: "exponential"
```

### 配置3：消融实验（混合模式）

```yaml
hierarchical:
  enabled: true

  # 标准配置
  step_level_estimator: "gae"
  token_level_estimator: "gae"
  reward_assignment: "last_token"

  # 混合原始rewards
  use_original_rewards: true
  mixing_alpha: 0.5  # 50% hierarchical + 50% original
```

## 监控指标

### Step-level指标

```python
metrics = {
    # Step values
    "hierarchical/step_value/mean": step_values.mean(),
    "hierarchical/step_value/std": step_values.std(),
    "hierarchical/step_value/max": step_values.max(),
    "hierarchical/step_value/min": step_values.min(),

    # Step advantages
    "hierarchical/step_advantage/mean": step_advantages.mean(),
    "hierarchical/step_advantage/std": step_advantages.std(),

    # Step returns
    "hierarchical/step_return/mean": step_returns.mean(),
    "hierarchical/step_return/std": step_returns.std(),

    # Correlation
    "hierarchical/step_value_reward_corr": correlation(step_values, env_rewards),
}
```

### Token-level指标

```python
metrics = {
    # Intrinsic rewards
    "hierarchical/intrinsic_reward/mean": intrinsic_rewards.mean(),
    "hierarchical/intrinsic_reward/std": intrinsic_rewards.std(),

    # Token advantages
    "hierarchical/token_advantage/mean": token_advantages.mean(),
    "hierarchical/token_advantage/std": token_advantages.std(),

    # Comparison with original
    "hierarchical/advantage_diff": (token_advantages - original_advantages).abs().mean(),
}
```

### 层级关系指标

```python
metrics = {
    # 上下层对齐度
    "hierarchical/layer_alignment": compute_alignment(step_returns, token_advantages),

    # Reward分配效率
    "hierarchical/assignment_entropy": compute_entropy(token_intrinsic_rewards),
}
```

## 实现优先级

### Phase 1: 最小可行实现（MVP）

1. **配置系统**
   - `HierarchicalRLConfig` dataclass
   - 配置验证

2. **核心计算**
   - Step value提取（仅last_token）
   - Step-level GAE
   - Reward分配（仅last_token）
   - Token-level GAE

3. **Pipeline集成**
   - 修改`agentic_pipeline.py`的train_step
   - 基本的指标记录

### Phase 2: 完善功能

1. **多种estimator**
   - N-step returns
   - REINFORCE variants

2. **多种分配策略**
   - uniform, exponential, value_weighted

3. **多种value提取方式**
   - mean_tokens, max_tokens

### Phase 3: 高级特性

1. **混合模式**
   - 原始rewards混合
   - 消融实验支持

2. **调试工具**
   - 详细日志
   - 可视化

3. **性能优化**
   - 批量计算优化
   - 内存优化

## 理论支持

这个设计基于以下经典Hierarchical RL方法：

1. **Feudal Networks (FuN)**
   - Worker接收Manager的intrinsic rewards
   - 不同时间尺度的操作

2. **MAXQ**
   - Value function decomposition
   - Pseudo-rewards for subtasks

3. **Options Framework**
   - Temporal abstraction
   - Sub-policies with intrinsic rewards

## 预期效果

启用Hierarchical RL后，预期能看到：

1. **更好的信用分配**
   - Step-level GAE提供更准确的长期价值估计
   - Token知道自己对整体目标的贡献

2. **更低的方差**
   - 上层使用bootstrap减少方差
   - 下层接收更稳定的训练信号

3. **更快的收敛**
   - 层级结构加速学习
   - 特别是在长序列任务上

4. **更好的Off-policy性能**
   - Bootstrap提供更准确的value估计
   - 与replay buffer配合更好

## 总结

这个Hierarchical RL设计实现了真正的两层强化学习：
- **上层**：评估环境中的长期价值
- **下层**：接收上层指导，优化生成策略
- **连接**：通过intrinsic rewards传递价值信号

相比原有实现，这是一个重大改进，让n-step returns和GAE真正发挥作用！