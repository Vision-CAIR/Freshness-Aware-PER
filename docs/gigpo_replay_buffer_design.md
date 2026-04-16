# GiGPO + Replay Buffer 大规模训练适配设计

## 1. 设计目标

### 1.1 核心目标

为大规模LLM Agent训练设计GiGPO与Replay Buffer的深度集成方案，实现：

1. **完整保持GiGPO的两层分组结构**
   - Episode-level: traj_group_id分组比较
   - Step-level: state_hash分组比较

2. **高样本效率**
   - 通过Replay Buffer重用历史数据
   - train_steps_per_env_step > 1

3. **训练稳定性**
   - Group-aware采样保证分组有效性
   - Off-policy校正防止策略崩溃

### 1.2 设计原则

- **Group-First**: 以traj_group为核心组织单位
- **Episode-Complete**: 保证episode完整性
- **State-Aware**: 最大化state分组效果
- **Scalable**: 支持百万级样本容量

---

## 2. GiGPO算法回顾

### 2.1 两层分组结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GiGPO Two-Level Structure                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Level 1: Episode-Level (Outer Layer)                                │   │
│  │                                                                     │   │
│  │   traj_group_id = "FrozenLake_0_42_1234"                           │   │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                  │   │
│  │   │ traj_0  │ │ traj_1  │ │ traj_2  │ │ traj_3  │  (group_size=4) │   │
│  │   │ score=1 │ │ score=0 │ │ score=1 │ │ score=0 │                  │   │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘                  │   │
│  │                      ↓                                              │   │
│  │   episode_reward = normalize([1, 0, 1, 0]) = [0.5, -0.5, 0.5, -0.5]│   │
│  │   → 成功的trajectory获得正向优势                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Level 2: Step-Level (Inner Layer)                                   │   │
│  │                                                                     │   │
│  │   state_hash = "abc123" (同一个state)                              │   │
│  │   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐            │   │
│  │   │ traj_0/step_2 │ │ traj_1/step_3 │ │ traj_2/step_2 │            │   │
│  │   │ action=Right  │ │ action=Down   │ │ action=Right  │            │   │
│  │   │ G_t = 0.95    │ │ G_t = 0.0     │ │ G_t = 0.90    │            │   │
│  │   └───────────────┘ └───────────────┘ └───────────────┘            │   │
│  │                      ↓                                              │   │
│  │   step_reward = normalize([0.95, 0.0, 0.90]) = [0.4, -0.8, 0.4]   │   │
│  │   → 从同一state出发，选择Right比Down更好                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Final: response_level_rewards = w_ep × episode_reward + w_st × step_reward│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 GiGPO的数据依赖

| 字段 | 用途 | 分组层级 |
|------|------|---------|
| `traj_group_id` | Episode-level分组 | 外层 |
| `traj_id` | 标识单个trajectory | 外层 |
| `episode_scores` | Episode总奖励 | 外层 |
| `state_hash` | Step-level分组 | 内层 |
| `step` | Step索引，计算折扣回报 | 内层 |
| `step_scores` | 单步即时奖励 | 内层 |

### 2.3 关键约束

1. **Episode完整性**: 计算`step_rewards`（折扣回报）需要同一episode的所有step
2. **Group完整性**: Episode-level比较需要同一group的多个episode
3. **State重叠**: Step-level比较需要相同state的多个样本

---

## 3. 数据结构设计

### 3.1 三层索引结构

```python
class StepReplayBuffer:
    def __init__(self, capacity: int, ...):
        # 核心存储
        self.steps: deque[StepEntry] = deque(maxlen=capacity)

        # ============================================
        # 三层索引结构 (Group -> Episode -> Step)
        # ============================================

        # Layer 1: Group Index
        # traj_group_id -> set of traj_ids belonging to this group
        self._group_index: Dict[str, Set[str]] = {}

        # Layer 2: Episode Index (已有)
        # traj_id -> {step_idx: buffer_idx}
        self._episode_index: Dict[str, Dict[int, int]] = {}

        # Layer 3: State Index
        # state_hash -> list of buffer_indices with this state
        self._state_index: Dict[str, List[int]] = {}

        # 反向映射
        self._buffer_to_episode: Dict[int, Tuple[str, int]] = {}  # buffer_idx -> (traj_id, step)
        self._traj_to_group: Dict[str, str] = {}  # traj_id -> traj_group_id
```

### 3.2 索引维护

```python
def _index_step_entry(self, buffer_idx: int, entry: StepEntry):
    """添加step时更新所有索引"""
    traj_id = entry.traj_id
    traj_group_id = entry.traj_group_id
    state_hash = entry.state_hash
    step = entry.step

    # Update Group Index
    if traj_group_id not in self._group_index:
        self._group_index[traj_group_id] = set()
    self._group_index[traj_group_id].add(traj_id)
    self._traj_to_group[traj_id] = traj_group_id

    # Update Episode Index
    if traj_id not in self._episode_index:
        self._episode_index[traj_id] = {}
    self._episode_index[traj_id][step] = buffer_idx

    # Update State Index
    if state_hash not in self._state_index:
        self._state_index[state_hash] = []
    self._state_index[state_hash].append(buffer_idx)

    # Update reverse mapping
    self._buffer_to_episode[buffer_idx] = (traj_id, step)

def _deindex_step_entry(self, buffer_idx: int, entry: StepEntry):
    """移除step时更新所有索引（eviction时调用）"""
    traj_id = entry.traj_id
    traj_group_id = entry.traj_group_id
    state_hash = entry.state_hash
    step = entry.step

    # Remove from Episode Index
    if traj_id in self._episode_index:
        if step in self._episode_index[traj_id]:
            del self._episode_index[traj_id][step]
        # If episode is empty, remove it and update group index
        if not self._episode_index[traj_id]:
            del self._episode_index[traj_id]
            # Remove from Group Index
            if traj_group_id in self._group_index:
                self._group_index[traj_group_id].discard(traj_id)
                if not self._group_index[traj_group_id]:
                    del self._group_index[traj_group_id]
            if traj_id in self._traj_to_group:
                del self._traj_to_group[traj_id]

    # Remove from State Index
    if state_hash in self._state_index:
        if buffer_idx in self._state_index[state_hash]:
            self._state_index[state_hash].remove(buffer_idx)
        if not self._state_index[state_hash]:
            del self._state_index[state_hash]

    # Remove from reverse mapping
    if buffer_idx in self._buffer_to_episode:
        del self._buffer_to_episode[buffer_idx]
```

### 3.3 Group完整性定义

```python
@dataclass
class GroupInfo:
    """Group的完整信息"""
    traj_group_id: str
    traj_ids: List[str]           # 属于这个group的所有traj_id
    complete_episodes: List[str]  # 完整的episode（有step 0且done=True）
    total_steps: int              # 总step数

    # 质量指标
    num_complete_episodes: int    # 完整episode数量
    state_diversity: int          # 不同state的数量
    avg_episode_length: float     # 平均episode长度

    @property
    def is_valid_for_gigpo(self) -> bool:
        """是否满足GiGPO的最低要求"""
        return self.num_complete_episodes >= 2  # 至少2个完整episode才能比较

def _analyze_group(self, traj_group_id: str) -> GroupInfo:
    """分析一个group的完整信息"""
    if traj_group_id not in self._group_index:
        return None

    traj_ids = list(self._group_index[traj_group_id])
    complete_episodes = []
    total_steps = 0
    all_states = set()

    buffer_list = list(self.steps)

    for traj_id in traj_ids:
        if traj_id not in self._episode_index:
            continue

        step_dict = self._episode_index[traj_id]
        total_steps += len(step_dict)

        # Check if episode is complete
        if 0 not in step_dict:
            continue

        sorted_steps = sorted(step_dict.keys())
        last_idx = step_dict[sorted_steps[-1]]

        if last_idx < len(buffer_list) and buffer_list[last_idx].done:
            complete_episodes.append(traj_id)

            # Collect states
            for step in sorted_steps:
                idx = step_dict[step]
                if idx < len(buffer_list):
                    all_states.add(buffer_list[idx].state_hash)

    return GroupInfo(
        traj_group_id=traj_group_id,
        traj_ids=traj_ids,
        complete_episodes=complete_episodes,
        total_steps=total_steps,
        num_complete_episodes=len(complete_episodes),
        state_diversity=len(all_states),
        avg_episode_length=total_steps / len(traj_ids) if traj_ids else 0
    )
```

---

## 4. Grouped Sampling 核心算法

### 4.1 采样流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Grouped Sampling Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 筛选有效Groups                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ valid_groups = [g for g in _group_index                             │   │
│  │                 if g.num_complete_episodes >= min_episodes_per_group]│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                             │
│  Step 2: 采样Groups (支持优先级)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ sampled_groups = priority_sample(valid_groups, num_groups)          │   │
│  │ 优先级可基于: reward, freshness, state_diversity                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                             │
│  Step 3: 每个Group内采样Episodes                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ for group in sampled_groups:                                        │   │
│  │     episodes = sample_episodes_from_group(group, episodes_per_group)│   │
│  │     # 优先选择有state重叠的episodes                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                             │
│  Step 4: 提取所有Steps并构建Batch                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ all_steps = flatten([get_episode_steps(ep) for ep in all_episodes]) │   │
│  │ batch = build_dataproto(all_steps)                                  │   │
│  │ batch.meta_info["group_boundaries"] = compute_boundaries()          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 核心采样方法

```python
def sample_for_gigpo(
    self,
    num_groups: int,
    episodes_per_group: int,
    min_episodes_per_group: int = 2,
    device: str = 'cpu',
    tokenizer: Optional[PreTrainedTokenizer] = None,
    sequence_length: int = 4096,
    group_priority: Literal["uniform", "reward", "freshness", "diversity"] = "uniform",
    episode_priority: Literal["uniform", "reward", "state_overlap"] = "state_overlap",
    compute_importance_weights: bool = False,
    importance_weight_beta: float = 0.4,
) -> Optional[Tuple[DataProto, Dict]]:
    """
    为GiGPO采样，保证Group结构完整性。

    Args:
        num_groups: 采样的group数量
        episodes_per_group: 每个group采样的episode数量
        min_episodes_per_group: group的最小完整episode数（过滤条件）
        group_priority: group采样优先级策略
            - "uniform": 均匀随机
            - "reward": 高奖励group优先
            - "freshness": 新鲜group优先
            - "diversity": state多样性高的group优先
        episode_priority: group内episode采样策略
            - "uniform": 均匀随机
            - "reward": 高奖励episode优先
            - "state_overlap": 优先选择有state重叠的episode组合

    Returns:
        Tuple of:
        - DataProto: 包含所有采样steps的batch
        - Dict: 元信息，包含group_boundaries, episode_boundaries等
    """

    # Step 1: 分析并筛选有效groups
    valid_groups = self._get_valid_groups_for_gigpo(min_episodes_per_group)

    if len(valid_groups) == 0:
        logger.warning("[GIGPO_SAMPLE] No valid groups found!")
        return None, {}

    # Step 2: 采样groups
    sampled_groups = self._sample_groups(
        valid_groups,
        num_groups,
        priority=group_priority
    )

    # Step 3: 每个group内采样episodes
    all_episodes = []
    group_boundaries = []  # 记录每个group在batch中的起始位置
    current_position = 0

    for group_info in sampled_groups:
        group_boundaries.append({
            'traj_group_id': group_info.traj_group_id,
            'start_idx': current_position,
            'episodes': []
        })

        # 采样这个group的episodes
        episodes = self._sample_episodes_from_group(
            group_info,
            episodes_per_group,
            priority=episode_priority
        )

        for episode in episodes:
            episode_start = current_position
            steps = self._get_episode_steps(episode['traj_id'])
            current_position += len(steps)

            group_boundaries[-1]['episodes'].append({
                'traj_id': episode['traj_id'],
                'start_idx': episode_start,
                'length': len(steps)
            })

            all_episodes.append({
                'traj_id': episode['traj_id'],
                'steps': steps
            })

        group_boundaries[-1]['end_idx'] = current_position

    # Step 4: 构建DataProto
    batch = self._build_gigpo_batch(
        all_episodes,
        device=device,
        tokenizer=tokenizer,
        sequence_length=sequence_length
    )

    # Step 5: 计算importance weights（可选）
    if compute_importance_weights:
        weights = self._compute_group_importance_weights(
            all_episodes,
            beta=importance_weight_beta
        )
        batch.batch["importance_weights"] = weights

    # 添加元信息
    batch.meta_info.update({
        "from_replay_buffer": True,
        "buffer_type": "step",
        "sampling_mode": "gigpo_grouped",
        "num_groups": len(sampled_groups),
        "episodes_per_group": episodes_per_group,
        "group_boundaries": group_boundaries,
        "total_episodes": len(all_episodes),
        "total_steps": current_position,
    })

    # 添加监控指标
    batch.meta_info["gigpo_metrics"] = self._compute_gigpo_sampling_metrics(
        sampled_groups, all_episodes
    )

    return batch, {"group_boundaries": group_boundaries}
```

### 4.3 Group采样策略

```python
def _sample_groups(
    self,
    valid_groups: List[GroupInfo],
    num_groups: int,
    priority: str = "uniform"
) -> List[GroupInfo]:
    """采样groups"""

    if len(valid_groups) <= num_groups:
        return valid_groups

    if priority == "uniform":
        return self.rng.sample(valid_groups, num_groups)

    elif priority == "reward":
        # 按group的平均episode reward排序
        def get_group_reward(g: GroupInfo) -> float:
            rewards = []
            for traj_id in g.complete_episodes:
                # 获取episode的reward
                if traj_id in self._episode_index:
                    first_step_idx = self._episode_index[traj_id].get(0)
                    if first_step_idx is not None and first_step_idx < len(self.steps):
                        rewards.append(self.steps[first_step_idx].episode_scores)
            return np.mean(rewards) if rewards else 0.0

        # 按reward排序，取top
        sorted_groups = sorted(valid_groups, key=get_group_reward, reverse=True)
        # 添加一些随机性，避免总是选同样的
        top_k = min(num_groups * 2, len(sorted_groups))
        return self.rng.sample(sorted_groups[:top_k], num_groups)

    elif priority == "freshness":
        # 按group的平均存储时间排序（越新越优先）
        def get_group_freshness(g: GroupInfo) -> float:
            ages = []
            for traj_id in g.complete_episodes:
                if traj_id in self._episode_index:
                    for step, idx in self._episode_index[traj_id].items():
                        if idx < len(self.steps):
                            ages.append(self.current_global_step - self.steps[idx].stored_at_step)
            return -np.mean(ages) if ages else float('-inf')  # 负数，越大越新

        sorted_groups = sorted(valid_groups, key=get_group_freshness, reverse=True)
        top_k = min(num_groups * 2, len(sorted_groups))
        return self.rng.sample(sorted_groups[:top_k], num_groups)

    elif priority == "diversity":
        # 按state多样性排序
        sorted_groups = sorted(valid_groups, key=lambda g: g.state_diversity, reverse=True)
        top_k = min(num_groups * 2, len(sorted_groups))
        return self.rng.sample(sorted_groups[:top_k], num_groups)

    else:
        raise ValueError(f"Unknown group priority: {priority}")
```

### 4.4 Episode采样策略（State-Overlap优化）

```python
def _sample_episodes_from_group(
    self,
    group_info: GroupInfo,
    num_episodes: int,
    priority: str = "state_overlap"
) -> List[Dict]:
    """从一个group内采样episodes，优化state重叠"""

    complete_episodes = group_info.complete_episodes

    if len(complete_episodes) <= num_episodes:
        return [{'traj_id': ep} for ep in complete_episodes]

    if priority == "uniform":
        selected = self.rng.sample(complete_episodes, num_episodes)
        return [{'traj_id': ep} for ep in selected]

    elif priority == "reward":
        # 按episode reward排序
        def get_episode_reward(traj_id: str) -> float:
            if traj_id in self._episode_index:
                first_step_idx = self._episode_index[traj_id].get(0)
                if first_step_idx is not None and first_step_idx < len(self.steps):
                    return self.steps[first_step_idx].episode_scores
            return 0.0

        sorted_episodes = sorted(complete_episodes, key=get_episode_reward, reverse=True)
        return [{'traj_id': ep} for ep in sorted_episodes[:num_episodes]]

    elif priority == "state_overlap":
        # 贪心选择state重叠最大化的episode组合
        return self._greedy_select_overlapping_episodes(
            complete_episodes,
            num_episodes
        )

    else:
        raise ValueError(f"Unknown episode priority: {priority}")

def _greedy_select_overlapping_episodes(
    self,
    episode_ids: List[str],
    num_episodes: int
) -> List[Dict]:
    """贪心选择state重叠最大化的episode组合"""

    # Step 1: 构建每个episode的state集合
    episode_states: Dict[str, Set[str]] = {}
    for traj_id in episode_ids:
        states = set()
        if traj_id in self._episode_index:
            for step, idx in self._episode_index[traj_id].items():
                if idx < len(self.steps):
                    states.add(self.steps[idx].state_hash)
        episode_states[traj_id] = states

    # Step 2: 贪心选择
    selected = []
    remaining = set(episode_ids)
    covered_states: Set[str] = set()

    # 第一个episode：选择state最多的
    first_ep = max(remaining, key=lambda ep: len(episode_states[ep]))
    selected.append({'traj_id': first_ep})
    covered_states.update(episode_states[first_ep])
    remaining.remove(first_ep)

    # 后续episode：选择与已选episode state重叠最多的
    while len(selected) < num_episodes and remaining:
        best_ep = None
        best_overlap = -1

        for ep in remaining:
            overlap = len(episode_states[ep] & covered_states)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ep = ep

        if best_ep is None:
            # 如果没有重叠，随机选一个
            best_ep = self.rng.choice(list(remaining))

        selected.append({'traj_id': best_ep})
        covered_states.update(episode_states[best_ep])
        remaining.remove(best_ep)

    return selected
```

---

## 5. Pipeline集成

### 5.1 配置扩展

```python
# agentic_config.py

@dataclass
class GiGPOReplayConfig:
    """GiGPO专用Replay Buffer配置"""

    # 采样粒度
    num_groups: int = field(
        default=16,
        metadata={"help": "每次采样的group数量"}
    )
    episodes_per_group: int = field(
        default=4,
        metadata={"help": "每个group采样的episode数量"}
    )
    min_episodes_per_group: int = field(
        default=2,
        metadata={"help": "group的最小完整episode数（过滤条件）"}
    )

    # 优先级策略
    group_priority: Literal["uniform", "reward", "freshness", "diversity"] = field(
        default="freshness",
        metadata={"help": "Group采样优先级: uniform/reward/freshness/diversity"}
    )
    episode_priority: Literal["uniform", "reward", "state_overlap"] = field(
        default="state_overlap",
        metadata={"help": "Group内episode采样优先级: uniform/reward/state_overlap"}
    )

    # Off-policy控制
    use_importance_weights: bool = field(
        default=False,
        metadata={"help": "是否使用importance sampling weights"}
    )
    importance_beta: float = field(
        default=0.4,
        metadata={"help": "Importance weights的beta参数"}
    )

    # 新鲜度控制
    max_sample_age: int = field(
        default=-1,  # -1表示不限制
        metadata={"help": "最大样本年龄（global_step差值），超过则不采样"}
    )

@dataclass
class ReplayConfig:
    # ... 现有字段 ...

    # GiGPO专用配置
    gigpo: GiGPOReplayConfig = field(
        default_factory=GiGPOReplayConfig,
        metadata={"help": "GiGPO专用replay配置"}
    )
```

### 5.2 Pipeline采样逻辑修改

```python
# agentic_pipeline.py

def _sample_from_replay_buffer(self) -> Optional[DataProto]:
    """从replay buffer采样"""

    if self.pipeline_config.adv_estimator == "gigpo":
        # GiGPO专用采样
        return self._sample_for_gigpo()
    else:
        # 原有逻辑
        return self._sample_generic()

def _sample_for_gigpo(self) -> Optional[DataProto]:
    """GiGPO专用采样逻辑"""

    gigpo_config = self.pipeline_config.replay.gigpo

    # 调用新的采样方法
    batch, meta = self.replay_buffer.sample_for_gigpo(
        num_groups=gigpo_config.num_groups,
        episodes_per_group=gigpo_config.episodes_per_group,
        min_episodes_per_group=gigpo_config.min_episodes_per_group,
        device=self.device,
        tokenizer=self.tokenizer,
        sequence_length=self.pipeline_config.sequence_length,
        group_priority=gigpo_config.group_priority,
        episode_priority=gigpo_config.episode_priority,
        compute_importance_weights=gigpo_config.use_importance_weights,
        importance_weight_beta=gigpo_config.importance_beta,
    )

    if batch is None:
        logger.warning("[GIGPO] Failed to sample from replay buffer")
        return None

    # 记录采样指标
    if "gigpo_metrics" in batch.meta_info:
        self.metrics.update({
            f"replay/gigpo/{k}": v
            for k, v in batch.meta_info["gigpo_metrics"].items()
        })

    return batch
```

### 5.3 监控指标

```python
def _compute_gigpo_sampling_metrics(
    self,
    sampled_groups: List[GroupInfo],
    all_episodes: List[Dict]
) -> Dict[str, float]:
    """计算GiGPO采样的质量指标"""

    metrics = {}

    # Group-level指标
    metrics["num_groups_sampled"] = len(sampled_groups)
    metrics["avg_episodes_per_group"] = np.mean([
        g.num_complete_episodes for g in sampled_groups
    ])
    metrics["avg_state_diversity_per_group"] = np.mean([
        g.state_diversity for g in sampled_groups
    ])

    # Episode-level指标
    metrics["total_episodes"] = len(all_episodes)
    total_steps = sum(len(ep['steps']) for ep in all_episodes)
    metrics["total_steps"] = total_steps
    metrics["avg_episode_length"] = total_steps / len(all_episodes) if all_episodes else 0

    # State overlap指标
    all_states = []
    for ep in all_episodes:
        ep_states = set()
        for step in ep['steps']:
            ep_states.add(step.state_hash)
        all_states.append(ep_states)

    if len(all_states) >= 2:
        # 计算平均state重叠率
        overlaps = []
        for i in range(len(all_states)):
            for j in range(i + 1, len(all_states)):
                if all_states[i] and all_states[j]:
                    overlap = len(all_states[i] & all_states[j]) / min(len(all_states[i]), len(all_states[j]))
                    overlaps.append(overlap)
        metrics["avg_state_overlap_ratio"] = np.mean(overlaps) if overlaps else 0.0

    # 有效state分组数（至少2个样本的state）
    state_counts = {}
    for ep in all_episodes:
        for step in ep['steps']:
            state_counts[step.state_hash] = state_counts.get(step.state_hash, 0) + 1

    effective_state_groups = sum(1 for count in state_counts.values() if count >= 2)
    metrics["effective_state_groups"] = effective_state_groups
    metrics["avg_samples_per_state"] = np.mean(list(state_counts.values())) if state_counts else 0

    return metrics
```

---

## 6. 配置示例

### 6.1 完整配置文件

```yaml
# frozen_lake_gigpo_replay_grouped.yaml

# ============================================
# 算法配置
# ============================================
adv_estimator: "gigpo"
batch_adjust_mode: "copy"
step_reward_weight: 1.0
episode_reward_weight: 1.0
step_reward_gamma: 0.95

# KL控制（重要！）
use_kl_loss: true
kl_loss_coef: 0.01

# Advantage处理
advantage_clip: 20
whiten_advantages: false

# Reward normalization
reward_normalization:
  grouping: traj_group_id
  method: mean

# ============================================
# 训练配置
# ============================================
max_steps: 10000
rollout_batch_size: 1024
sequence_length: 1024

actor_train:
  training_args:
    learning_rate: 1.0e-6
    warmup_steps: 100
    per_device_train_batch_size: 16
    gradient_accumulation_steps: 8

# ============================================
# 环境配置
# ============================================
train_env_manager:
  num_env_groups: 128
  group_size: 8              # 每个prompt生成8个trajectory
  tags: [FrozenLake]
  num_groups_partition: [128]

# ============================================
# Replay Buffer配置 (GiGPO Grouped Sampling)
# ============================================
replay:
  enabled: true
  capacity: 500000           # 50万step容量

  # 训练节奏
  train_steps_per_env_step: 2
  minibatch_size: 1024

  # 优先级
  priority_function: "reward_fresh"
  priority_exponent: 0.6
  age_decay: 1000.0

  # GiGPO专用配置
  gigpo:
    # 采样粒度
    num_groups: 16           # 每次采样16个groups
    episodes_per_group: 4    # 每个group采样4个episodes
    min_episodes_per_group: 2

    # 优先级策略
    group_priority: "freshness"     # 新鲜group优先
    episode_priority: "state_overlap"  # state重叠优化

    # Off-policy控制
    use_importance_weights: false
    max_sample_age: 5000     # 最多采样5000步前的数据
```

### 6.2 不同规模的配置推荐

| 规模 | num_groups | episodes_per_group | group_size | capacity |
|------|-----------|-------------------|------------|----------|
| 小型 (单机8卡) | 8 | 4 | 4 | 100K |
| 中型 (16卡) | 16 | 4 | 8 | 500K |
| 大型 (32卡+) | 32 | 8 | 8 | 1M+ |

---

## 7. 实现计划

### Phase 1: 索引结构 (1天)

**目标**: 实现三层索引结构

**任务**:
- [ ] 添加 `_group_index` 和 `_state_index`
- [ ] 实现 `_index_step_entry` 和 `_deindex_step_entry`
- [ ] 修改 `push_from_dataproto` 使用新索引
- [ ] 修改 eviction 逻辑维护索引

**测试**:
```python
def test_index_consistency():
    """测试索引一致性"""
    buffer = StepReplayBuffer(capacity=1000)
    # 添加数据
    # 验证三层索引一致
    # 触发eviction
    # 验证索引正确更新
```

### Phase 2: GroupInfo分析 (0.5天)

**目标**: 实现Group分析功能

**任务**:
- [ ] 实现 `GroupInfo` dataclass
- [ ] 实现 `_analyze_group` 方法
- [ ] 实现 `_get_valid_groups_for_gigpo` 方法

### Phase 3: 采样方法 (2天)

**目标**: 实现完整的GiGPO采样

**任务**:
- [ ] 实现 `_sample_groups` (多种优先级策略)
- [ ] 实现 `_sample_episodes_from_group` (state_overlap优化)
- [ ] 实现 `_greedy_select_overlapping_episodes`
- [ ] 实现 `sample_for_gigpo` 主方法
- [ ] 实现 `_build_gigpo_batch`

### Phase 4: Pipeline集成 (1天)

**目标**: 集成到agentic_pipeline

**任务**:
- [ ] 添加 `GiGPOReplayConfig` 配置类
- [ ] 修改 `_sample_from_replay_buffer` 逻辑
- [ ] 添加监控指标
- [ ] 创建示例配置文件

### Phase 5: 测试与优化 (1-2天)

**目标**: 验证正确性和性能

**任务**:
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试 (大容量buffer)
- [ ] 对比实验

---

## 8. 预期效果

### 8.1 样本效率提升

| 配置 | 样本效率 | 说明 |
|------|---------|------|
| GiGPO (on-policy) | 1x | 基准 |
| GiGPO + Replay (grouped, train_steps=2) | 1.8-2x | 每条数据训练2次 |
| GiGPO + Replay (grouped, train_steps=4) | 2.5-3x | 更激进的重用 |

### 8.2 训练稳定性

通过以下机制保证稳定性:
- `group_priority: freshness` - 优先使用新鲜数据
- `episode_priority: state_overlap` - 保证state分组有效
- `use_kl_loss: true` - KL约束防止策略崩溃
- `max_sample_age` - 限制数据年龄

### 8.3 监控指标示例

```
replay/gigpo/num_groups_sampled: 16
replay/gigpo/avg_episodes_per_group: 5.2
replay/gigpo/avg_state_diversity_per_group: 8.3
replay/gigpo/total_episodes: 83
replay/gigpo/total_steps: 498
replay/gigpo/avg_state_overlap_ratio: 0.42
replay/gigpo/effective_state_groups: 156
replay/gigpo/avg_samples_per_state: 3.2
```

---

## 9. 附录

### A. 与现有方法的兼容性

| 现有方法 | 兼容性 | 说明 |
|---------|-------|------|
| `sample_for_training` | 保持 | step-level采样，用于其他算法 |
| `sample_episodes_for_hierarchical` | 保持 | episode-level采样，被GiGPO采样复用 |
| Priority functions | 兼容 | group_priority使用现有优先级函数 |
| N-step returns | 独立 | GiGPO使用自己的折扣回报计算 |

### B. 参考资料

1. [GiGPO Paper](https://arxiv.org/abs/2505.10978)
2. [ROLL官方GiGPO配置](examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_gigpo.yaml)
3. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
