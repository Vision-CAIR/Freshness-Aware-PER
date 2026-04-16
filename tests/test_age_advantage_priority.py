"""
Unit tests for Age-Aware Advantage-Based Priority Replay Buffer.

Tests the two-stage priority system:
1. Initial priority: reward-based
2. Updated priority: advantage-based
3. Age decay: freshness weighting
"""

import pytest
import numpy as np
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.agentic.replay_buffer.step_buffer import StepReplayBuffer, StepEntry
from roll.agentic.replay_buffer.trajectory_buffer import TrajectoryReplayBuffer, TrajectoryEntry


def create_mock_step_dataproto(batch_size=4, seq_len=64, rewards=None):
    """Create a mock DataProto for step-level data."""
    if rewards is None:
        rewards = np.random.randn(batch_size)

    batch = DataProto()
    batch.batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        "response_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "prompt_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "scores": torch.zeros(batch_size, seq_len),
        "penalty": torch.zeros(batch_size),
        "behavior_log_probs": torch.zeros(batch_size, seq_len - 1),
    }, batch_size=[batch_size])

    # Set response tokens and rewards
    for i in range(batch_size):
        response_start = seq_len // 2
        batch.batch["response_mask"][i, response_start:] = True
        batch.batch["prompt_mask"][i, response_start:] = False
        batch.batch["scores"][i, response_start:] = rewards[i] / (seq_len - response_start)

    batch.non_tensor_batch = {
        "env_ids": np.array([f"env_{i}" for i in range(batch_size)], dtype=object),
        "group_ids": np.array([f"group_{i}" for i in range(batch_size)], dtype=object),
        "messages_list": np.array([[]] * batch_size, dtype=object),
        "tags": np.array(["test"] * batch_size, dtype=object),
        "frames": np.array([[]] * batch_size, dtype=object),
        "step_scores": np.array([[]] * batch_size, dtype=object),
        "episode_scores": np.array([[]] * batch_size, dtype=object),
        "traj_group_id": np.array([f"traj_group_{i}" for i in range(batch_size)], dtype=object),
        "traj_id": np.array([f"traj_{i}" for i in range(batch_size)], dtype=object),
        "state_hash": np.array([f"hash_{i}" for i in range(batch_size)], dtype=object),
        "step": np.array([0] * batch_size, dtype=object),
    }

    return batch


def test_initial_reward_based_priority():
    """Test that initial priorities are based on rewards."""
    from roll.agentic.replay_buffer.priority_functions import reward_priority

    buffer = StepReplayBuffer(
        capacity=100,
        priority_fn=reward_priority,
        age_decay=1000.0,
        use_advantage_priority=True
    )

    # Push samples with different rewards
    rewards = np.array([1.0, 5.0, 0.1, 10.0])
    batch = create_mock_step_dataproto(batch_size=4, rewards=rewards)
    buffer.push_from_dataproto(batch, global_step=0)

    # Check that priorities reflect reward magnitudes
    assert len(buffer.steps) == 4
    priorities = [step.priority for step in buffer.steps]

    # Higher rewards should have higher initial priorities
    assert priorities[3] > priorities[1] > priorities[0] > priorities[2]
    print(f"✓ Initial priorities based on rewards: {priorities}")


def test_advantage_based_priority_update():
    """Test that priorities are updated with advantages after training."""
    from roll.agentic.replay_buffer.priority_functions import uniform_priority

    buffer = StepReplayBuffer(
        capacity=100,
        priority_fn=uniform_priority,
        age_decay=1000.0,
        use_advantage_priority=True
    )

    # Push samples
    batch = create_mock_step_dataproto(batch_size=4)
    buffer.push_from_dataproto(batch, global_step=0)

    # Create mock advantages
    advantages = np.array([0.1, 2.0, 0.5, 5.0])

    # Update priorities with advantages
    indices = [0, 1, 2, 3]
    buffer.update_priorities(indices, advantages, current_global_step=10)

    # Check that priorities now reflect advantage magnitudes
    updated_priorities = [buffer.steps[i].priority for i in indices]

    # Higher advantages should have higher priorities
    assert updated_priorities[3] > updated_priorities[1] > updated_priorities[2] > updated_priorities[0]
    print(f"✓ Priorities updated with advantages: {updated_priorities}")


def test_age_decay_freshness_weight():
    """Test that age decay properly weights priorities by freshness."""
    from roll.agentic.replay_buffer.priority_functions import uniform_priority

    age_decay = 100.0
    buffer = StepReplayBuffer(
        capacity=100,
        priority_fn=uniform_priority,
        age_decay=age_decay,
        use_advantage_priority=True
    )

    # Push samples at different times
    batch1 = create_mock_step_dataproto(batch_size=2)
    buffer.push_from_dataproto(batch1, global_step=0)  # Old samples

    batch2 = create_mock_step_dataproto(batch_size=2)
    buffer.push_from_dataproto(batch2, global_step=200)  # Fresh samples

    # Update all with same intrinsic priority
    intrinsic_priority = 1.0
    indices = [0, 1, 2, 3]
    buffer.update_priorities(
        indices,
        np.array([intrinsic_priority] * 4),
        current_global_step=200
    )

    # Compute effective priorities
    effective_0 = buffer.get_effective_priority(0, current_global_step=200)  # age=200
    effective_2 = buffer.get_effective_priority(2, current_global_step=200)  # age=0

    # Fresh sample should have higher effective priority
    assert effective_2 > effective_0

    # Check decay formula: exp(-age / age_decay)
    expected_freshness_0 = np.exp(-200 / age_decay)
    expected_effective_0 = intrinsic_priority * expected_freshness_0

    assert np.isclose(effective_0, expected_effective_0, atol=1e-5)
    print(f"✓ Age decay working: old={effective_0:.4f}, fresh={effective_2:.4f}")


def test_compute_advantage_priorities_static_method():
    """Test the static method for computing advantage-based priorities."""
    # Create mock batch with advantages
    batch_size = 4
    seq_len = 64

    batch = DataProto()
    advantages = torch.tensor([
        [0.1] * 32 + [0.0] * 32,  # Low advantage
        [2.0] * 32 + [0.0] * 32,  # High advantage
        [0.5] * 32 + [0.0] * 32,  # Medium advantage
        [5.0] * 32 + [0.0] * 32,  # Very high advantage
    ])
    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    response_mask[:, :32] = True  # First 32 tokens are response

    batch.batch = TensorDict({
        "advantages": advantages,
        "response_mask": response_mask
    }, batch_size=[batch_size])

    # Compute priorities
    priorities = StepReplayBuffer.compute_advantage_priorities(batch)

    # Check shape
    assert priorities.shape == (batch_size,)

    # Check values match mean absolute advantages
    expected = np.array([0.1, 2.0, 0.5, 5.0])
    assert np.allclose(priorities, expected, atol=1e-5)

    # Check ordering
    assert priorities[3] > priorities[1] > priorities[2] > priorities[0]
    print(f"✓ Advantage priorities computed correctly: {priorities}")


def test_two_stage_priority_workflow():
    """Test the complete two-stage priority workflow: reward → advantage."""
    from roll.agentic.replay_buffer.priority_functions import reward_priority

    buffer = StepReplayBuffer(
        capacity=100,
        priority_fn=reward_priority,
        age_decay=1000.0,
        use_advantage_priority=True
    )

    # Stage 1: Push with reward-based initial priorities
    rewards = np.array([1.0, 0.1, 5.0, 2.0])
    batch = create_mock_step_dataproto(batch_size=4, rewards=rewards)
    buffer.push_from_dataproto(batch, global_step=0)

    initial_priorities = [step.priority for step in buffer.steps]
    print(f"Initial priorities (reward-based): {initial_priorities}")

    # Verify reward-based ordering
    assert initial_priorities[2] > initial_priorities[3] > initial_priorities[0] > initial_priorities[1]

    # Stage 2: Update with advantage-based priorities
    advantages = np.array([5.0, 2.0, 0.1, 1.0])  # Different ordering than rewards!
    buffer.update_priorities([0, 1, 2, 3], advantages, current_global_step=10)

    updated_priorities = [step.priority for step in buffer.steps]
    print(f"Updated priorities (advantage-based): {updated_priorities}")

    # Verify advantage-based ordering
    assert updated_priorities[0] > updated_priorities[1] > updated_priorities[3] > updated_priorities[2]

    # Verify priorities changed from initial
    assert not np.array_equal(initial_priorities, updated_priorities)
    print("✓ Two-stage priority workflow working correctly")


def test_trajectory_buffer_age_advantage():
    """Test age-advantage priority for trajectory buffer."""
    from roll.agentic.replay_buffer.priority_functions import reward_priority

    buffer = TrajectoryReplayBuffer(
        capacity=100,
        priority_fn=reward_priority,
        age_decay=500.0,
        use_advantage_priority=True
    )

    # Push trajectories
    rewards = np.array([1.0, 5.0])
    batch = create_mock_step_dataproto(batch_size=2, rewards=rewards)
    buffer.push_from_dataproto(batch, global_step=0)

    # Check initial priorities
    initial_priorities = [traj.priority for traj in buffer.trajectories]
    assert initial_priorities[1] > initial_priorities[0]

    # Update with advantages
    advantages = np.array([3.0, 2.0])  # Reversed ordering
    buffer.update_priorities([0, 1], advantages, current_global_step=100)

    updated_priorities = [traj.priority for traj in buffer.trajectories]
    assert updated_priorities[0] > updated_priorities[1]  # Ordering changed

    # Check age decay
    effective_0 = buffer.get_effective_priority(0, current_global_step=100)
    effective_1 = buffer.get_effective_priority(1, current_global_step=100)

    # Both have same age, so effective priority should match intrinsic
    assert effective_0 > effective_1
    print("✓ Trajectory buffer age-advantage working correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Age-Aware Advantage-Based Priority Replay Buffer")
    print("=" * 60)

    test_initial_reward_based_priority()
    test_advantage_based_priority_update()
    test_age_decay_freshness_weight()
    test_compute_advantage_priorities_static_method()
    test_two_stage_priority_workflow()
    test_trajectory_buffer_age_advantage()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
