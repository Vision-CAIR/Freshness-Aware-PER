"""
Unit tests for N-Step Returns and Episode Index in StepReplayBuffer.

Tests cover:
1. Episode Index creation and maintenance
2. Episode Index eviction during buffer wrap-around
3. N-Step Returns computation
4. Incomplete episode handling
5. GAE computation (basic tests)
"""

import pytest
import numpy as np
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.agentic.replay_buffer.step_buffer import StepReplayBuffer


def create_test_batch(traj_id: str, steps: list, rewards: list, batch_size: int = None):
    """
    Create a test DataProto batch with specified trajectory and rewards.

    Args:
        traj_id: Trajectory ID
        steps: List of step numbers (e.g., [0, 1, 2, 3, 4])
        rewards: List of rewards for each step
        batch_size: Number of samples (defaults to len(steps))

    Returns:
        DataProto batch
    """
    if batch_size is None:
        batch_size = len(steps)

    assert len(steps) == len(rewards), "steps and rewards must have same length"

    # Create dummy data
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    response_mask[:, seq_len // 2:] = True  # Second half is response
    prompt_mask = ~response_mask
    penalty = torch.zeros(batch_size)

    # Scores with rewards on response tokens
    scores = torch.zeros((batch_size, seq_len))
    for i, reward in enumerate(rewards):
        # Distribute reward evenly across response tokens
        num_response_tokens = response_mask[i].sum().item()
        scores[i, response_mask[i]] = reward / num_response_tokens

    batch = DataProto()
    batch.batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "prompt_mask": prompt_mask,
        "scores": scores,
        "penalty": penalty,
    }, batch_size=[batch_size])

    batch.non_tensor_batch = {
        "env_ids": np.array([f"env_{traj_id}_{s}" for s in steps], dtype=object),
        "group_ids": np.array([f"group_{traj_id}" for _ in steps], dtype=object),
        "messages_list": np.array([[{"role": "user", "content": "test"}] for _ in steps], dtype=object),
        "tags": np.array(["test" for _ in steps], dtype=object),
        "frames": np.array([[] for _ in steps], dtype=object),
        "step_scores": np.array([[r] for r in rewards], dtype=object),
        "episode_scores": np.array([[sum(rewards)] for _ in steps], dtype=object),
        "traj_group_id": np.array([f"traj_group_{traj_id}" for _ in steps], dtype=object),
        "traj_id": np.array([traj_id for _ in steps], dtype=object),
        "state_hash": np.array([f"hash_{traj_id}_{s}" for s in steps], dtype=object),
        "step": np.array(steps, dtype=object),
    }

    return batch


class TestEpisodeIndex:
    """Test Episode Index creation and maintenance."""

    def test_episode_index_basic(self):
        """Test basic episode index creation."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5)

        # Push a 5-step episode
        batch = create_test_batch(traj_id="ep1", steps=[0, 1, 2, 3, 4], rewards=[1.0, 2.0, 3.0, 4.0, 5.0])
        buffer.push_from_dataproto(batch, global_step=0)

        # Verify episode index
        assert "ep1" in buffer._episode_index
        assert len(buffer._episode_index["ep1"]) == 5
        assert buffer._episode_index["ep1"][0] == 0
        assert buffer._episode_index["ep1"][4] == 4

        # Verify reverse mapping
        assert buffer._buffer_to_episode[0] == ("ep1", 0)
        assert buffer._buffer_to_episode[4] == ("ep1", 4)

    def test_episode_index_multiple_episodes(self):
        """Test episode index with multiple episodes."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5)

        # Push episode 1
        batch1 = create_test_batch(traj_id="ep1", steps=[0, 1, 2], rewards=[1.0, 2.0, 3.0])
        buffer.push_from_dataproto(batch1, global_step=0)

        # Push episode 2
        batch2 = create_test_batch(traj_id="ep2", steps=[0, 1, 2, 3], rewards=[4.0, 5.0, 6.0, 7.0])
        buffer.push_from_dataproto(batch2, global_step=0)

        # Verify both episodes exist
        assert "ep1" in buffer._episode_index
        assert "ep2" in buffer._episode_index
        assert len(buffer._episode_index["ep1"]) == 3
        assert len(buffer._episode_index["ep2"]) == 4

        # Verify indices
        assert buffer._episode_index["ep1"][0] == 0
        assert buffer._episode_index["ep1"][2] == 2
        assert buffer._episode_index["ep2"][0] == 3
        assert buffer._episode_index["ep2"][3] == 6

    def test_episode_index_wrap_around(self):
        """Test episode index cleanup during buffer wrap-around."""
        buffer = StepReplayBuffer(capacity=10, enable_nstep=True, n_step=5)

        # Fill buffer with episode 1 (5 steps)
        batch1 = create_test_batch(traj_id="ep1", steps=[0, 1, 2, 3, 4], rewards=[1.0] * 5)
        buffer.push_from_dataproto(batch1, global_step=0)

        # Add episode 2 (5 steps)
        batch2 = create_test_batch(traj_id="ep2", steps=[0, 1, 2, 3, 4], rewards=[2.0] * 5)
        buffer.push_from_dataproto(batch2, global_step=0)

        # Verify buffer is full
        assert len(buffer.steps) == 10

        # Push episode 3 (5 steps) - should evict ep1
        batch3 = create_test_batch(traj_id="ep3", steps=[0, 1, 2, 3, 4], rewards=[3.0] * 5)
        buffer.push_from_dataproto(batch3, global_step=0)

        # Verify ep1 is evicted
        assert "ep1" not in buffer._episode_index
        assert "ep2" in buffer._episode_index
        assert "ep3" in buffer._episode_index

        # Verify reverse mapping cleanup
        for i in range(5):
            assert i not in buffer._buffer_to_episode or buffer._buffer_to_episode[i][0] == "ep3"


class TestNStepIndices:
    """Test N-Step index traversal."""

    def test_get_nstep_indices_complete(self):
        """Test get_nstep_indices with complete episode."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5)

        # Push a 10-step episode
        batch = create_test_batch(
            traj_id="ep1",
            steps=list(range(10)),
            rewards=[1.0] * 10
        )
        buffer.push_from_dataproto(batch, global_step=0)

        # Get 5-step sequence from beginning
        indices, complete = buffer.get_nstep_indices(start_idx=0, n_step=5)

        assert complete is True
        assert indices == [0, 1, 2, 3, 4]

        # Get 5-step sequence from middle
        indices, complete = buffer.get_nstep_indices(start_idx=3, n_step=5)

        assert complete is True
        assert indices == [3, 4, 5, 6, 7]

    def test_get_nstep_indices_incomplete(self):
        """Test get_nstep_indices with incomplete episode."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5)

        # Push a 3-step episode
        batch = create_test_batch(traj_id="ep1", steps=[0, 1, 2], rewards=[1.0, 2.0, 3.0])
        buffer.push_from_dataproto(batch, global_step=0)

        # Try to get 5-step sequence (only 3 available)
        indices, complete = buffer.get_nstep_indices(start_idx=0, n_step=5)

        assert complete is False
        assert len(indices) == 3
        assert indices == [0, 1, 2]

        # From last step, only 1 step available
        indices, complete = buffer.get_nstep_indices(start_idx=2, n_step=5)

        assert complete is False
        assert len(indices) == 1
        assert indices == [2]

    def test_get_nstep_indices_boundary(self):
        """Test get_nstep_indices at episode boundaries."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5)

        # Push two episodes
        batch1 = create_test_batch(traj_id="ep1", steps=[0, 1, 2], rewards=[1.0, 2.0, 3.0])
        buffer.push_from_dataproto(batch1, global_step=0)

        batch2 = create_test_batch(traj_id="ep2", steps=[0, 1, 2], rewards=[4.0, 5.0, 6.0])
        buffer.push_from_dataproto(batch2, global_step=0)

        # Try to get 5-step sequence from ep1's last step
        # Should stop at episode boundary
        indices, complete = buffer.get_nstep_indices(start_idx=2, n_step=5)

        assert complete is False
        assert len(indices) == 1  # Only last step of ep1
        assert indices == [2]


class TestNStepReturns:
    """Test N-Step Returns computation."""

    def test_compute_nstep_returns_complete(self):
        """Test n-step return computation with complete sequences."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=3, gamma=0.99)

        # Create episode with known rewards: [1, 2, 3, 4, 5]
        batch = create_test_batch(
            traj_id="ep1",
            steps=[0, 1, 2, 3, 4],
            rewards=[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        buffer.push_from_dataproto(batch, global_step=0)

        # Compute 3-step return from step 0
        # Expected: 1 + 0.99*2 + 0.99^2*3 = 1 + 1.98 + 2.9403 = 5.9203
        returns, completeness = buffer.compute_nstep_returns(
            sampled_indices=[0],
            n_step=3,
            gamma=0.99
        )

        assert completeness[0] is True
        assert np.isclose(returns[0], 5.9203, atol=1e-3)

        # Compute 3-step return from step 1
        # Expected: 2 + 0.99*3 + 0.99^2*4 = 2 + 2.97 + 3.9204 = 8.8904
        returns, completeness = buffer.compute_nstep_returns(
            sampled_indices=[1],
            n_step=3,
            gamma=0.99
        )

        assert completeness[0] is True
        assert np.isclose(returns[0], 8.8904, atol=1e-3)

    def test_compute_nstep_returns_incomplete(self):
        """Test n-step return with incomplete episodes."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5, gamma=0.99)

        # Push only 2 steps of an episode
        batch = create_test_batch(traj_id="ep1", steps=[0, 1], rewards=[1.0, 2.0])
        buffer.push_from_dataproto(batch, global_step=0)

        # Try to compute 5-step return (only 2 steps available)
        returns, completeness = buffer.compute_nstep_returns(
            sampled_indices=[0],
            n_step=5,
            gamma=0.99
        )

        # Should only use available steps: 1 + 0.99*2 = 2.98
        assert completeness[0] is False
        assert np.isclose(returns[0], 2.98, atol=1e-3)

    def test_compute_nstep_returns_with_bootstrap(self):
        """Test n-step return with bootstrap values."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=3, gamma=0.99)

        # Create episode: [1, 2, 3, 4, 5]
        batch = create_test_batch(
            traj_id="ep1",
            steps=[0, 1, 2, 3, 4],
            rewards=[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        buffer.push_from_dataproto(batch, global_step=0)

        # Provide bootstrap value
        bootstrap_values = np.array([10.0])

        # Compute 3-step return with bootstrap from step 0
        # Expected: 1 + 0.99*2 + 0.99^2*3 + 0.99^3*10
        #         = 1 + 1.98 + 2.9403 + 9.7030 = 15.6233
        returns, completeness = buffer.compute_nstep_returns(
            sampled_indices=[0],
            n_step=3,
            gamma=0.99,
            bootstrap_values=bootstrap_values
        )

        assert completeness[0] is True
        assert np.isclose(returns[0], 15.6233, atol=1e-3)

    def test_compute_nstep_returns_batch(self):
        """Test batch n-step return computation."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=3, gamma=0.99)

        # Push two episodes
        batch1 = create_test_batch(traj_id="ep1", steps=[0, 1, 2, 3], rewards=[1.0, 2.0, 3.0, 4.0])
        buffer.push_from_dataproto(batch1, global_step=0)

        batch2 = create_test_batch(traj_id="ep2", steps=[0, 1, 2], rewards=[5.0, 6.0, 7.0])
        buffer.push_from_dataproto(batch2, global_step=0)

        # Compute n-step returns for multiple samples
        returns, completeness = buffer.compute_nstep_returns(
            sampled_indices=[0, 4],  # First step of ep1 and ep2
            n_step=3,
            gamma=0.99
        )

        assert len(returns) == 2
        assert len(completeness) == 2

        # Verify both are complete
        assert completeness[0] is True
        assert completeness[1] is True

        # Verify returns
        # ep1 step 0: 1 + 0.99*2 + 0.99^2*3 = 5.9203
        # ep2 step 0: 5 + 0.99*6 + 0.99^2*7 = 17.8607
        assert np.isclose(returns[0], 5.9203, atol=1e-3)
        assert np.isclose(returns[1], 17.8607, atol=1e-3)


class TestIntegration:
    """Integration tests with sample_for_training."""

    def test_sample_with_nstep_enabled(self):
        """Test sampling with n-step enabled."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5, gamma=0.99)

        # Push episode
        batch = create_test_batch(
            traj_id="ep1",
            steps=list(range(10)),
            rewards=[1.0] * 10
        )
        buffer.push_from_dataproto(batch, global_step=0)

        # Sample
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataproto, indices = buffer.sample_for_training(
            batch_size=5,
            device='cpu',
            tokenizer=tokenizer,
            sequence_length=128
        )

        # Verify n-step returns are computed
        assert "nstep_returns" in dataproto.batch
        assert "nstep_completeness" in dataproto.batch
        assert dataproto.batch["nstep_returns"].shape == (5,)
        assert "nstep_complete_ratio" in dataproto.meta_info

    def test_sample_with_nstep_disabled(self):
        """Test sampling with n-step disabled."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=False)

        # Push episode
        batch = create_test_batch(
            traj_id="ep1",
            steps=list(range(10)),
            rewards=[1.0] * 10
        )
        buffer.push_from_dataproto(batch, global_step=0)

        # Sample
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataproto, indices = buffer.sample_for_training(
            batch_size=5,
            device='cpu',
            tokenizer=tokenizer,
            sequence_length=128
        )

        # Verify n-step returns are NOT computed
        assert "nstep_returns" not in dataproto.batch
        assert "nstep_completeness" not in dataproto.batch


class TestBufferStats:
    """Test buffer statistics with episode index."""

    def test_episode_index_stats(self):
        """Test episode index statistics."""
        buffer = StepReplayBuffer(capacity=100, enable_nstep=True, n_step=5)

        # Push multiple episodes
        for i in range(5):
            steps = list(range(i + 1))  # Variable length episodes
            rewards = [1.0] * (i + 1)
            batch = create_test_batch(traj_id=f"ep{i}", steps=steps, rewards=rewards)
            buffer.push_from_dataproto(batch, global_step=0)

        # Get stats
        stats = buffer.get_stats()

        # Verify episode index stats
        assert "episode_index/num_episodes" in stats
        assert stats["episode_index/num_episodes"] == 5
        assert "episode_index/avg_episode_length" in stats
        # Average: (1 + 2 + 3 + 4 + 5) / 5 = 3
        assert np.isclose(stats["episode_index/avg_episode_length"], 3.0)
        assert "episode_index/total_indexed_steps" in stats
        assert stats["episode_index/total_indexed_steps"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
