"""
Priority Calculation Functions for Replay Buffer

This module provides pluggable priority functions for prioritized experience replay.
Each function takes a trajectory/step entry and returns a priority value.

Usage:
    from roll.agentic.replay_buffer.priority_functions import uniform_priority, reward_priority

    buffer = TrajectoryReplayBuffer(
        capacity=10000,
        priority_fn=reward_priority  # Use reward-based priority
    )
"""

import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .trajectory_buffer import TrajectoryEntry
    from .step_buffer import StepEntry


def uniform_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    **kwargs
) -> float:
    """
    Uniform priority - all samples have equal priority.

    This is equivalent to uniform random sampling and is the default behavior.

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        **kwargs: Additional arguments (ignored)

    Returns:
        Priority value of 1.0 for all samples
    """
    return 1.0


def reward_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    epsilon: float = 1e-6,
    **kwargs
) -> float:
    """
    Reward-based priority - samples with higher absolute rewards have higher priority.

    This prioritizes trajectories that had significant outcomes (positive or negative).
    Useful for focusing on high-impact experiences.

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        epsilon: Small constant to ensure non-zero priority
        **kwargs: Additional arguments (ignored)

    Returns:
        Priority based on |reward| + epsilon
    """
    # Extract reward from scores
    # For trajectory: last token's score is episode reward
    # For step: sum of scores
    if hasattr(entry, 'episode_length'):  # TrajectoryEntry
        # Get the last valid score (episode reward)
        mask = entry.attention_mask.astype(bool)
        valid_scores = entry.scores[mask]
        if len(valid_scores) > 0:
            reward = float(valid_scores[-1])
        else:
            reward = 0.0
    else:  # StepEntry
        reward = float(entry.scores.sum())

    # Priority = |reward| + epsilon
    priority = abs(reward) + epsilon
    return float(priority)


def recency_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    alpha: float = 0.001,
    **kwargs
) -> float:
    """
    Recency-based priority - newer samples have higher priority.

    Uses exponential decay to prefer recent experiences over old ones.
    This can help the agent focus on more relevant recent data.

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        alpha: Decay rate (higher = faster decay)
        **kwargs: Additional arguments (ignored)

    Returns:
        Priority based on exp(-alpha * age)
    """
    age = global_step - entry.stored_at_step
    # Exponential decay: e^(-alpha * age)
    priority = np.exp(-alpha * age)
    return float(priority)


def combined_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    reward_weight: float = 0.5,
    recency_weight: float = 0.5,
    epsilon: float = 1e-6,
    alpha: float = 0.001,
    **kwargs
) -> float:
    """
    Combined priority - weighted combination of multiple strategies.

    Combines reward-based and recency-based priorities to balance
    focusing on high-reward experiences and recent experiences.

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        reward_weight: Weight for reward component [0, 1]
        recency_weight: Weight for recency component [0, 1]
        epsilon: Small constant for reward priority
        alpha: Decay rate for recency priority
        **kwargs: Additional arguments (ignored)

    Returns:
        Weighted combination of reward and recency priorities
    """
    # Normalize weights
    total_weight = reward_weight + recency_weight
    if total_weight > 0:
        reward_weight = reward_weight / total_weight
        recency_weight = recency_weight / total_weight
    else:
        reward_weight = 0.5
        recency_weight = 0.5

    # Calculate component priorities
    reward_p = reward_priority(entry, global_step, epsilon=epsilon)
    recency_p = recency_priority(entry, global_step, alpha=alpha)

    # Normalize each component to [0, 1] range before combining
    # This prevents one component from dominating
    reward_p_norm = reward_p / (reward_p + 1.0)  # Normalize to [0, 1)
    recency_p_norm = recency_p  # Already in [0, 1]

    # Weighted combination
    priority = reward_weight * reward_p_norm + recency_weight * recency_p_norm
    return float(priority)


def advantage_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    epsilon: float = 1e-6,
    **kwargs
) -> float:
    """
    Advantage-based priority - samples with higher advantages have higher priority.

    Requires advantages to be computed and passed via kwargs.
    Falls back to uniform priority if advantages not available.

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        epsilon: Small constant to ensure non-zero priority
        **kwargs: Must contain 'advantages' key with advantage values

    Returns:
        Priority based on |advantage| + epsilon, or 1.0 if not available
    """
    advantages = kwargs.get('advantages', None)

    if advantages is None:
        # Fallback to uniform if advantages not provided
        return 1.0

    # Use mean absolute advantage as priority
    if isinstance(advantages, np.ndarray):
        adv_value = float(np.abs(advantages).mean())
    else:
        adv_value = float(abs(advantages))

    priority = adv_value + epsilon
    return float(priority)


def td_error_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    epsilon: float = 1e-6,
    **kwargs
) -> float:
    """
    TD-error based priority - samples with higher TD errors have higher priority.

    This is similar to Prioritized Experience Replay (PER).
    Requires TD errors to be computed and passed via kwargs.
    Falls back to uniform priority if TD errors not available.

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        epsilon: Small constant to ensure non-zero priority
        **kwargs: Must contain 'td_error' key with TD error values

    Returns:
        Priority based on |TD-error| + epsilon, or 1.0 if not available
    """
    td_error = kwargs.get('td_error', None)

    if td_error is None:
        # Fallback to uniform if TD error not provided
        return 1.0

    # Use absolute TD error as priority
    if isinstance(td_error, np.ndarray):
        td_value = float(np.abs(td_error).mean())
    else:
        td_value = float(abs(td_error))

    priority = td_value + epsilon
    return float(priority)


def length_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    prefer_long: bool = True,
    epsilon: float = 1e-6,
    **kwargs
) -> float:
    """
    Length-based priority - prioritize based on trajectory/step length.

    Can prefer either long trajectories (more information) or short trajectories
    (potentially more efficient episodes).

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        prefer_long: If True, longer sequences have higher priority
        epsilon: Small constant to ensure non-zero priority
        **kwargs: Additional arguments (ignored)

    Returns:
        Priority based on sequence length
    """
    # Get episode length
    if hasattr(entry, 'episode_length'):  # TrajectoryEntry
        length = entry.episode_length
    else:  # StepEntry - use attention mask to get length
        length = int(entry.attention_mask.sum())

    if prefer_long:
        priority = float(length) + epsilon
    else:
        # Inverse length - shorter episodes have higher priority
        priority = 1.0 / (float(length) + epsilon)

    return float(priority)


def reward_fresh_priority(
    entry: Union["TrajectoryEntry", "StepEntry"],
    global_step: int,
    epsilon: float = 1e-6,
    age_decay: float = 500.0,
    **kwargs
) -> float:
    """
    Reward-Fresh priority - combines reward-based priority with age decay.

    This is our custom extension of standard PER (Prioritized Experience Replay).
    It addresses two key issues in off-policy LLM RL:
    1. High-reward samples are more informative for learning
    2. Fresher samples have less policy drift (closer to current policy)

    Formula:
        priority = (|reward| + epsilon) * exp(-age / age_decay)

    This ensures:
    - High-reward fresh samples get highest priority
    - Old samples get deprioritized regardless of reward
    - Zero-reward samples still have priority based on freshness

    Args:
        entry: Trajectory or step entry
        global_step: Current global training step
        epsilon: Small constant to ensure non-zero priority
        age_decay: Decay constant for age weighting (default 500.0)
            - Smaller values = faster decay (stronger preference for fresh samples)
            - age_decay=500: samples half-life ~346 steps
            - age_decay=1000: samples half-life ~693 steps
        **kwargs: Additional arguments (ignored)

    Returns:
        Priority based on |reward| * freshness_weight + epsilon

    Reference:
        - Fedus et al., "Revisiting Fundamentals of Experience Replay", ICML 2020
        - Schaul et al., "Prioritized Experience Replay", ICLR 2016
    """
    # Calculate reward component
    if hasattr(entry, 'episode_length'):  # TrajectoryEntry
        # Get the last valid score (episode reward)
        mask = entry.attention_mask.astype(bool)
        valid_scores = entry.scores[mask]
        if len(valid_scores) > 0:
            reward = float(valid_scores[-1])
        else:
            reward = 0.0
    else:  # StepEntry
        reward = float(entry.scores.sum())

    reward_component = abs(reward) + epsilon

    # Calculate age decay component
    age = max(0, global_step - entry.stored_at_step)
    freshness_weight = np.exp(-age / age_decay)

    # Combined priority: reward * freshness
    priority = reward_component * freshness_weight

    return float(priority)


# Special marker functions for deterministic sampling
# These are not actual priority functions but sampling strategies
def lifo_priority(entry, global_step, **kwargs) -> float:
    """
    LIFO (Last-In-First-Out) marker - deterministic sampling of newest entries.

    This is a special sampling strategy used in Echo mode for near-on-policy training.
    Returns uniform priority (1.0) as actual sampling is handled deterministically.

    Note: This is not a true priority function. The buffer will use deterministic
    LIFO sampling when this function is selected.
    """
    return 1.0


def fifo_priority(entry, global_step, **kwargs) -> float:
    """
    FIFO (First-In-First-Out) marker - deterministic sampling of oldest entries.

    Useful for ensuring all data is used before being evicted from the buffer.
    Returns uniform priority (1.0) as actual sampling is handled deterministically.

    Note: This is not a true priority function. The buffer will use deterministic
    FIFO sampling when this function is selected.
    """
    return 1.0


# Registry of all available priority functions
PRIORITY_FUNCTIONS = {
    # Deterministic sampling strategies (no weighted sampling)
    "uniform": uniform_priority,  # Random uniform sampling (standard)
    "lifo": lifo_priority,        # Last-In-First-Out (Echo mode, near-on-policy)
    "fifo": fifo_priority,        # First-In-First-Out (ensure old data usage)

    # Weighted sampling strategies (use priority_alpha)
    "reward": reward_priority,           # Priority based on |reward|
    "recency": recency_priority,         # Priority based on age (exponential decay)
    "combined": combined_priority,       # Combination of reward + recency
    "advantage": advantage_priority,     # Priority based on |advantage| (requires computation)
    "td_error": td_error_priority,       # Priority based on |TD-error| (standard PER)
    "length": length_priority,           # Priority based on sequence length
    "reward_fresh": reward_fresh_priority,  # Reward × age_decay (our custom PER extension)
}

# Mapping: priority_function -> update_metric
# This ensures consistency: the same signal used for initial priority is used for updates
PRIORITY_UPDATE_METRIC = {
    # No update needed (deterministic or auto-decaying)
    "uniform": None,
    "lifo": None,
    "fifo": None,
    "recency": None,   # Age decays automatically
    "length": None,    # Length doesn't change

    # Update with the same metric after training
    "reward": "reward",
    "advantage": "advantage",
    "td_error": "td_error",
    "combined": "reward",  # Only reward part needs update, recency auto-decays
    "reward_fresh": "reward",  # Reward part needs update, age decay is automatic
}


def get_update_metric(priority_function: str) -> str:
    """
    Get the update metric for a given priority function.

    Args:
        priority_function: Name of the priority function

    Returns:
        Update metric name, or None if no update needed
    """
    return PRIORITY_UPDATE_METRIC.get(priority_function.lower(), None)


def get_priority_function(name: str):
    """
    Get a priority function by name.

    Args:
        name: Name of the priority function

    Returns:
        Priority function callable

    Raises:
        ValueError: If priority function name is unknown
    """
    name = name.lower()
    if name not in PRIORITY_FUNCTIONS:
        available = ", ".join(PRIORITY_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown priority function: {name}. "
            f"Available functions: {available}"
        )
    return PRIORITY_FUNCTIONS[name]
