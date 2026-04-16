import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

from omegaconf import DictConfig

from roll.agentic.env import REGISTERED_ENV_CONFIGS
from roll.configs.base_config import BaseConfig
from roll.configs.worker_config import WorkerConfig
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from roll.pipeline.agentic.hierarchical_config import HierarchicalRLConfig

logger = get_logger()


@dataclass
class RewardNormalizationConfig:
    grouping: str = field(default="state", metadata={"help": "state / batch / inductive"})
    method: str = field(default="identity", metadata={"help": "asym_clip / identity / mean_std"})

@dataclass
class LLMProxyConfig:
    proxy_type: str = field(default="policy", metadata={"help": "llm proxy type: [policy, openai, random]."})
    proxy_config: Dict = field(default_factory=dict, metadata={"help": "llm proxy config."})


@dataclass
class VTraceConfig:
    """
    Configuration for V-trace advantage estimation.
    V-trace is an off-policy correction algorithm that uses truncated importance
    sampling to handle the difference between behavior and target policies.
    """
    rho_bar: float = field(
        default=1.0,
        metadata={
            "help": "Truncation threshold for importance sampling ratio ρ (rho). "
                   "Default 1.0 means truncate to [0, 1]. Higher values allow more off-policy correction."
        }
    )
    c_bar: float = field(
        default=1.0,
        metadata={
            "help": "Truncation threshold for trace coefficient c. "
                   "Default 1.0 means truncate to [0, 1]. Controls the speed of value function propagation."
        }
    )


@dataclass
class OffPolicyMonitorConfig:
    """
    Configuration for off-policy monitoring.
    This is independent of replay buffer and can be used in various scenarios:
    - Async training (policy drift between actor and trainer)
    - Multiple gradient updates (policy changes during training)
    - Replay buffer training (off-policy data)
    - External data loading
    """
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable off-policy monitoring for all training batches."}
    )

    # Behavior policy log probs computation
    save_behavior_log_probs: bool = field(
        default=True,
        metadata={"help": "Whether to compute and save behavior policy log probs after rollout."}
    )

    # Monitoring frequency
    monitor_fresh_batch: bool = field(
        default=True,
        metadata={"help": "Whether to monitor off-policy metrics for fresh rollout batches."}
    )
    monitor_replay_batch: bool = field(
        default=True,
        metadata={"help": "Whether to monitor off-policy metrics for replay buffer batches."}
    )
    monitor_interval: int = field(
        default=1,
        metadata={"help": "Monitor every N training steps (1 = every step)."}
    )


@dataclass
class ReplayConfig:
    enabled: bool = field(default=False, metadata={"help": "Enable replay buffer for agentic training."})
    capacity: int = field(default=1000000, metadata={"help": "Max number of step transitions stored in replay buffer (1M steps, not episodes)."})
    min_size: int = field(default=2000, metadata={"help": "Minimum step transitions before sampling is allowed (recommended: 2x batch_size)."})
    train_steps_per_env_step: int = field(default=1, metadata={"help": "Number of training steps per rollout step when replay is enabled."})

    # Batch size configuration
    minibatch_size: int = field(default=128, metadata={"help": "Legacy compatibility. Use use_rollout_batch_size=True instead."})

    # Step-based textual buffer configuration
    use_rollout_batch_size: bool = field(
        default=True,
        metadata={"help": "Use rollout_batch_size for sampling instead of minibatch_size (recommended for step-based buffer)."}
    )

    # Storage mode configuration
    storage_mode: Literal["hybrid", "text_only", "tokens_only"] = field(
        default="hybrid",
        metadata={"help": "Storage mode: 'hybrid' (text+tokens), 'text_only' (pure text), 'tokens_only' (pure tokens)"}
    )

    # Manager type detection (for advanced users)
    source_manager_type: str = field(
        default="auto",
        metadata={"help": "Source env_manager type: 'auto' (detect), 'trajectory' (TrajEnvManager), 'step' (StepEnvManager)"}
    )

    lazy_tokenization: bool = field(
        default=False,
        metadata={"help": "If True, tokenize only during sampling (memory efficient, but slower sampling)"}
    )

    # Replay buffer integration settings
    replay_ratio: float = field(
        default=0.5,
        metadata={"help": "Ratio of replay data in each training batch (0.0-1.0). 0.5 means 50% replay, 50% fresh data"}
    )

    # Sampling mode settings
    sampling_mode: Literal["trajectory", "step"] = field(
        default="trajectory",
        metadata={"help": "Sampling mode for replay: 'trajectory' samples full episodes; 'step' samples per-step items"}
    )
    steps_per_episode: int = field(
        default=1,
        metadata={"help": "When sampling_mode='step', number of steps to sample per episode in one minibatch."}
    )

    # Unified sampling method across buffer strategies
    sample_method: Literal["uniform", "fifo", "lifo"] = field(
        default="uniform",
        metadata={
            "help": "Sampling method: uniform (random), fifo (oldest first by timestamp), lifo (newest first by timestamp)."
        }
    )

    # Grouped sampling for GRPO-style training (K candidates per prompt)
    candidates_per_group: int = field(
        default=1,
        metadata={"help": "Number of candidates per group (K). Use K>1 for GRPO; K=1 for reinforce/GAE."}
    )
    group_sampling: Literal["uniform", "fifo", "lifo"] = field(
        default="uniform",
        metadata={"help": "Group selection strategy when candidates_per_group>1."}
    )
    min_groups: int = field(
        default=0,
        metadata={"help": "Warmup threshold in groups when using grouped sampling (optional)."}
    )
    # Train updates use only replay minibatches (fresh rollouts are only pushed into buffer)
    train_from_replay_only: bool = field(
        default=False,
        metadata={"help": "If true, skip main on-policy update and train only from replay minibatches."}
    )

    # N-Step Returns Configuration
    enable_nstep: bool = field(
        default=False,
        metadata={"help": "Enable n-step returns computation for replay buffer."}
    )
    n_step: int = field(
        default=5,
        metadata={
            "help": "Number of steps for n-step returns (step-level, not token-level). "
                    "This refers to environment interaction steps."
        }
    )
    nstep_gamma: float = field(
        default=0.99,
        metadata={
            "help": "Discount factor for step-level n-step returns (γ_step). "
                    "This is used for outer-layer (step-level) discounting, "
                    "separate from token-level gamma."
        }
    )
    use_nstep_in_advantage: bool = field(
        default=False,
        metadata={
            "help": "Whether to use n-step returns as outer-layer reward in advantage computation. "
                    "If True, n-step returns replace single-step response_level_rewards when "
                    "expanding to token-level, but inner-layer token-level computation remains unchanged."
        }
    )
    use_bootstrap: bool = field(
        default=False,
        metadata={"help": "Whether to use critic values for bootstrapping in n-step returns."}
    )

    # GAE Configuration (Advanced)
    enable_gae: bool = field(
        default=False,
        metadata={"help": "Enable Generalized Advantage Estimation (GAE) for replay buffer."}
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "GAE lambda parameter for exponential smoothing of TD errors."}
    )
    gae_horizon: int = field(
        default=20,
        metadata={"help": "Truncation horizon for GAE computation (limits lookback)."}
    )

    # ==========================================================================
    # Priority Configuration (Prioritized Experience Replay)
    # ==========================================================================
    # priority_function determines BOTH initial priority AND update metric.
    # This ensures consistency - no more split between "initial" and "update" priorities.
    #
    # Available functions:
    #   - "uniform": Equal priority, random sampling (default)
    #   - "lifo": Last-In-First-Out, newest samples first (Echo mode)
    #   - "fifo": First-In-First-Out, oldest samples first
    #   - "reward": Priority = |reward|, updated with reward after training
    #   - "advantage": Priority = |advantage|, updated after training
    #   - "td_error": Priority = |TD-error|, standard PER
    #   - "recency": Priority decays with age (no update needed)
    #   - "reward_fresh": Priority = |reward| × exp(-age/age_decay), our custom PER extension
    #                     Combines reward-based priority with age decay for LLM RL
    # ==========================================================================
    priority_function: str = field(
        default="uniform",
        metadata={
            "help": "Priority function for sampling. Determines both initial priority and update metric. "
                    "Options: uniform, lifo, fifo, reward, advantage, td_error, recency, reward_fresh."
        }
    )
    priority_exponent: float = field(
        default=0.6,
        metadata={
            "help": "Priority exponent (alpha in PER). 0.0 = uniform sampling, 1.0 = full prioritization."
        }
    )
    importance_sampling_correction: bool = field(
        default=False,
        metadata={
            "help": "Apply importance sampling correction to compensate for non-uniform sampling bias."
        }
    )
    importance_beta: float = field(
        default=0.4,
        metadata={
            "help": "Importance sampling exponent (beta in PER). Should anneal from 0.4 to 1.0 during training."
        }
    )
    enable_age_decay: bool = field(
        default=False,
        metadata={
            "help": "Enable age-based freshness weighting. When False (default), uses standard PER. "
                    "When True, older samples get lower effective priority."
        }
    )
    age_decay: float = field(
        default=1000.0,
        metadata={
            "help": "Age decay constant for freshness weighting (only used if enable_age_decay=True). "
                    "Effective priority = priority * exp(-age / age_decay). "
                    "Smaller values = stronger preference for fresh samples."
        }
    )
    refresh_interval: int = field(
        default=1,
        metadata={
            "help": "Interval (in training steps) for refreshing age decay of ALL samples in the buffer. "
                    "Only used when enable_age_decay=True. "
                    "Set to 1 for every step (recommended with async refresh), "
                    "or higher values (e.g., 5-10) to reduce CPU overhead. "
                    "The refresh is performed asynchronously during GPU training, "
                    "so setting to 1 typically has zero additional latency."
        }
    )
    eviction_strategy: Literal["fifo", "smart"] = field(
        default="fifo",
        metadata={
            "help": "Eviction strategy when buffer is full. 'fifo' (default) or 'smart'."
        }
    )

    # Off-policy Filtering Configuration
    enable_offpolicy_filter: bool = field(
        default=False,
        metadata={
            "help": "Enable off-policy filtering based on importance sampling ratio. "
                    "Filters out samples where the policy has diverged too far from behavior policy."
        }
    )
    ratio_clip_max: Optional[float] = field(
        default=3.0,
        metadata={
            "help": "Maximum allowed importance sampling ratio for filtering. "
                    "Samples with ratio > ratio_clip_max are filtered out."
        }
    )
    filter_mini_batch_size: int = field(
        default=32,
        metadata={
            "help": "Mini-batch size for filtering forward passes. "
                    "Smaller values reduce GPU memory usage but may increase total computation time."
        }
    )
    filter_max_attempts: int = field(
        default=20,
        metadata={
            "help": "Maximum number of mini-batches to sample during filtering. "
                    "Prevents infinite loops when filter rate is very high."
        }
    )
    filter_oversample_ratio: float = field(
        default=1.5,
        metadata={
            "help": "[DEPRECATED - using mini-batch instead] "
                    "Oversample ratio for filtering (e.g., 1.5 = sample 192 to get 128)."
        }
    )
    filter_min_acceptable_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Minimum acceptable batch size after filtering before supplementing with unfiltered samples. "
                    "If collected valid samples < this threshold, supplement with unfiltered samples to reach target_batch_size. "
                    "If None, defaults to target_batch_size // 2. "
                    "Set to 0 to always supplement when below target_batch_size."
        }
    )


@dataclass
class TrajectoryLogConfig:
    """Configuration for trajectory logging during training."""
    enabled: bool = field(
        default=True,
        metadata={"help": "Enable trajectory logging to separate JSONL file."}
    )
    save_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of trajectories to save (0.0-1.0). E.g., 0.1 = save 10% of trajectories."}
    )
    max_samples_per_step: int = field(
        default=20,
        metadata={"help": "Maximum number of trajectory samples to save per training step."}
    )
    filename: str = field(
        default="trajectory_samples.jsonl",
        metadata={"help": "Filename for trajectory log (will be saved in logging_dir)."}
    )


@dataclass
class EnvManagerConfig(WorkerConfig):
    llm_proxy: LLMProxyConfig = field(default_factory=LLMProxyConfig, metadata={"help": "llm proxy config."})
    num_env_groups: int = field(default=128, metadata={"help": "Number of environment groups during training."})
    group_size: int = field(
        default=1, metadata={"help": "Under the same group, the env config and env seed are ensured to be equal"}
    )
    tags: List[str] = field(default_factory=lambda: ["SimpleSokoban"], metadata={"help": "Environment tags."})
    num_groups_partition: List[int] = field(
        default_factory=lambda: [128],
        metadata={
            "help": "If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation"
        },
    )
    max_traj_per_env: int = field(
        default=-1, metadata={"help": "The maximum number of trajectories that each environment can rollout."}
    )
    format_penalty: float = field(default=0, metadata={"help": "Format penalty value."})
    worker_cls: Optional[str] = field(
        default="roll.pipeline.agentic.environment_worker.EnvironmentWorker",
        metadata={"help": "The class of the worker."},
    )
    max_env_num_per_worker: int = field(
        default=0,
        metadata={"help": "The maximum number of envs per worker. one env per thread."}
    )

    def __post_init__(self):
        """
        根据es config计算world_size
        """
        if self.max_env_num_per_worker <= 0:
            self.max_env_num_per_worker = self.num_env_groups * self.group_size
            logger.warning("all env in one worker by default, you can set max_env_num_per_worker to scale env.")
        logger.info(f"max_env_num_per_worker: {self.max_env_num_per_worker}")

        assert self.num_env_groups * self.group_size % self.max_env_num_per_worker == 0
        self.world_size = (self.num_env_groups * self.group_size + self.max_env_num_per_worker - 1) // self.max_env_num_per_worker
        self.env_configs: Optional[Dict[int, Dict[int, Dict]]] = None
        """
        worker_rank: 
            env_id:
                env_config
        """


@dataclass
class AgenticConfig(BaseConfig):
    # agentic related
    custom_envs: Dict[str, Any] = field(default_factory=dict, metadata={"help": "List of environment configurations."})
    train_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    val_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    render_save_dir: str = field(default=None, metadata={"help": "Directory to save rendered frames."})
    trajectory_log: TrajectoryLogConfig = field(
        default_factory=TrajectoryLogConfig,
        metadata={"help": "Trajectory logging configuration for debugging and analysis."}
    )
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig, metadata={"help": "Reward normalization configuration."}
    )
    offpolicy_monitor: OffPolicyMonitorConfig = field(
        default_factory=OffPolicyMonitorConfig,
        metadata={"help": "Off-policy monitoring configuration (independent of replay buffer)."}
    )
    replay: ReplayConfig = field(default_factory=ReplayConfig, metadata={"help": "Replay buffer configuration."})
    vtrace: VTraceConfig = field(default_factory=VTraceConfig, metadata={"help": "V-trace configuration for off-policy correction."})
    hierarchical: HierarchicalRLConfig = field(
        default_factory=HierarchicalRLConfig,
        metadata={"help": "Hierarchical RL configuration (only for StepEnvManager)."}
    )

    # role related
    pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory, if available."})
    reward_pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory for the reward model, if available."}
    )
    actor_train: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the actor's training role."}
    )
    actor_infer: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the actor's inference role."}
    )
    critic: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the critic's training role."}
    )
    reference: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the reference role."}
    )

    batch_adjust_mode: Literal["copy", "delete", "auto"] = field(
        default="copy", metadata={"help": "batch adjust mode: copy or delete"}
    )
    episode_reward_weight: float = field(default=1.0, metadata={"help": "Episode reward weight, used in GiGPO."})
    step_reward_weight: float = field(default=1.0, metadata={"help": "Step reward weight, used in GiGPO."})
    step_reward_gamma: float = field(default=0.95, metadata={"help": "Gamma parameter for step reward calculation"})

    # PPO related
    ppo_epochs: int = field(default=1, metadata={"help": "Number of optimisation epochs per batch of samples"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})
    l2: float = field(default=0.0, metadata={"help": "L2 regularization"})
    lambd: float = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})
    gamma: float = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    pg_clip: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"})
    value_clip: Optional[float] = field(
        default=None, metadata={"help": "Range for clipping values in loss calculation"}
    )
    kl_penalty: Literal["kl", "abs", "mse", "full"] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp, 'abs': abs(kl), 'mse': "
                    "mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"
        },
    )
    target_kl: Optional[float] = field(default=None, metadata={"help": "Target KL value for adaptive KL control"})
    init_kl_coef: float = field(
        default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"}
    )
    kl_horizon: int = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    use_reward_scaling: bool = field(default=False, metadata={"help": "Use reward scaling"})
    add_len_reward: bool = field(default=False)
    reward_clip: float = field(default=None, metadata={"help": "reward clip value."})
    use_reward_norm: bool = field(
        default=False, metadata={"help": "Use reward normalization. Only applicable if use_reward_scaling is True."}
    )
    whiten_rewards: bool = field(default=False, metadata={"help": "Whiten the rewards before compute advantages."})
    whiten_advantages: bool = field(default=False, metadata={"help": "Whiten the advantage."})
    advantage_clip: float = field(default=None, metadata={"help": "advantage_clip value"})
    adv_estimator: Literal["gae", "reinforce", "grpo", "gigpo", "vtrace"] = field(
        default="gae", metadata={"help": "advantage estimator: gae (GAE), vtrace (V-trace for off-policy)."}
    )
    reward_norm: Literal["batch", "group", "running", None] = field(
        default=None,
        metadata={
            "help": "Reward normalization type: 'batch' (normalize across batch), 'group' (normalize within prompt groups), 'running' (use running statistics)"
        },
    )
    reward_shift: bool = field(
        default=False, metadata={"help": "Only subtract mean without dividing by std during reward normalization"}
    )
    reward_scale: bool = field(
        default=False, metadata={"help": "Only divide by std without subtracting mean during reward normalization"}
    )
    add_token_level_kl: bool = field(default=False, metadata={"help": "Add token level kl penalty"})
    critic_warmup: int = field(
        default=0,
        metadata={"help": "Pre-training step for critic model"},
    )
    use_kl_loss: bool = field(default=False, metadata={"help": "Use kl loss"})
    kl_loss_coef: float = field(default=0, metadata={"help": "Loss coefficient for kl loss"})
    entropy_loss_coef: float = field(default=0, metadata={"help": "Loss coefficient for entropy loss"})
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"] = (
        field(default="seq-mean-token-sum", metadata={"help": "Loss aggregation mode"})
    )
    dual_clip_loss: bool = field(default=False, metadata={"help": "Use dual clip loss"})

    def __post_init__(self):
        BaseConfig.__post_init__(self)

        if (
            self.actor_train.model_args.model_name_or_path is None
            or self.actor_infer.model_args.model_name_or_path is None
            or self.reference.model_args.model_name_or_path is None
        ):
            self.actor_train.model_args.model_name_or_path = self.pretrain
            self.actor_infer.model_args.model_name_or_path = self.pretrain
            self.reference.model_args.model_name_or_path = self.pretrain

        if self.critic.model_args.model_name_or_path is None:
            self.critic.model_args.model_name_or_path = self.reward_pretrain

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.base_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"

        self.actor_train.training_args.output_dir = self.output_dir
        self.actor_infer.training_args.output_dir = self.output_dir
        self.critic.training_args.output_dir = self.output_dir

        self.actor_infer.name = "actor_infer"
        self.actor_train.name = "actor_train"
        self.reference.name = "reference"
        self.critic.name = "critic"
        self.train_env_manager.name = "train_env"
        self.val_env_manager.name = "val_env"

        self.actor_infer.generating_args.num_return_sequences = 1

        if self.render_save_dir:
            self.render_save_dir = os.path.join(
                self.render_save_dir, self.exp_name, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        logger.info(f"add timestamp to render_save_dir  {self.render_save_dir}")

        assert self.max_steps > 0, "max_steps must be greater than 0"

        self.train_env_manager.model_args.model_name_or_path = self.pretrain
        self.train_env_manager.generating_args = self.actor_infer.generating_args
        self.val_env_manager.model_args.model_name_or_path = self.pretrain
        self.val_env_manager.generating_args = self.actor_infer.generating_args
        self.custom_envs = DictConfig(self.custom_envs)
        self.make_env_configs(self.train_env_manager)
        self.make_env_configs(self.val_env_manager)

        train_env_num = self.train_env_manager.num_env_groups * self.train_env_manager.group_size
        traj_per_env = (self.rollout_batch_size + train_env_num - 1) // train_env_num
        if self.async_generation_ratio > 0:
            # force set max_traj_per_env when use async training
            self.train_env_manager.max_traj_per_env = traj_per_env
        elif self.train_env_manager.max_traj_per_env < 0:
            self.train_env_manager.max_traj_per_env = traj_per_env
        logger.info(f"train_env_manager.max_traj_per_env: {self.train_env_manager.max_traj_per_env}")
        assert self.train_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

        val_env_num = self.val_env_manager.num_env_groups * self.val_env_manager.group_size
        traj_per_env = (self.val_batch_size + val_env_num - 1) // val_env_num
        if self.val_env_manager.max_traj_per_env < 0:
            self.val_env_manager.max_traj_per_env = traj_per_env
        logger.info(f"val_env_manager.max_traj_per_env: {self.val_env_manager.max_traj_per_env}")
        assert self.val_env_manager.max_traj_per_env >= traj_per_env, f"max_traj_per_env must be >= {traj_per_env}"

    def make_env_configs(self, env_manager_config: EnvManagerConfig):
        # construct env configs
        env_configs = defaultdict(defaultdict)
        done_groups = 0
        env_manager_config.env_configs = {}
        group_seeds = {}
        max_env_num_per_worker = env_manager_config.max_env_num_per_worker
        for tag, n_group in zip(env_manager_config.tags, env_manager_config.num_groups_partition):
            for env_id in range(
                done_groups * env_manager_config.group_size, (done_groups + n_group) * env_manager_config.group_size
            ):
                cfg_template = self.custom_envs[tag]
                env_class = cfg_template.env_type
                max_tokens_per_step = cfg_template.max_tokens_per_step

                group_id = env_id // env_manager_config.group_size
                cfg_template.env_config["group_id"] = group_id
                cfg_template.env_config["group_size"] = env_manager_config.num_env_groups
                env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)

                if group_id not in group_seeds:
                    group_seeds[group_id] = random.randint(0, 1000000)
                entry = {}
                entry.update(cfg_template)
                entry.pop("env_config", None)
                entry.update({
                    "tag": tag,
                    "group_id": group_id,
                    "env_id": env_id,
                    "config": env_config,
                    "env_class": env_class,
                    "env_manager_cls": cfg_template.get("env_manager_cls", "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager"),
                    "group_seed": group_seeds[group_id],
                })
                worker_rank = env_id // max_env_num_per_worker
                env_configs[worker_rank][env_id] = entry
            done_groups += n_group
        assert done_groups == env_manager_config.num_env_groups
        env_manager_config.env_configs = env_configs

    def set_max_steps(self, max_steps: int):
        actor_backward_batch_size = (
                self.actor_train.training_args.per_device_train_batch_size
                * self.actor_train.training_args.gradient_accumulation_steps
        )
        critic_backward_batch_size = (
                self.critic.training_args.per_device_train_batch_size
                * self.critic.training_args.gradient_accumulation_steps
        )
        # 没有除dp_size，需要在分布式环境初始化后再除
        self.actor_train.training_args.max_steps = max_steps * (
                self.rollout_batch_size
                * self.actor_infer.generating_args.num_return_sequences
                * self.ppo_epochs
                // actor_backward_batch_size
        )
        self.critic.training_args.max_steps = max_steps * (
                self.rollout_batch_size
                * self.actor_infer.generating_args.num_return_sequences
                // critic_backward_batch_size
        )

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")
        logger.info(f"critic train max_steps without dp_size: {self.critic.training_args.max_steps}")
        self.max_steps = max_steps
