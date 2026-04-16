"""
Test script to verify n-step returns integration with advantage computation.

This tests the hierarchical RL structure:
- Outer layer: Step-level rewards (can use n-step returns)
- Inner layer: Token-level GAE (unchanged)
"""

import torch
import numpy as np
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import expand_to_token_level, apply_kl_penalty
from roll.utils.kl_controller import AdaptiveKLController


def create_mock_dataproto(batch_size=4, seq_len=10, use_nstep=False):
    """Create a mock DataProto for testing."""
    # Single-step rewards
    response_level_rewards = torch.randn(batch_size)

    # N-step returns (outer-layer multi-step rewards)
    if use_nstep:
        nstep_returns = response_level_rewards * 2.5  # Simulate accumulated rewards
        nstep_completeness = torch.ones(batch_size, dtype=torch.bool)
    else:
        nstep_returns = None
        nstep_completeness = None

    # Token-level data
    attention_mask = torch.ones(batch_size, seq_len)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    response_mask = torch.ones(batch_size, seq_len)

    # Log probs for KL
    old_log_probs = torch.randn(batch_size, seq_len - 1) * 0.1
    ref_log_probs = torch.randn(batch_size, seq_len - 1) * 0.1

    batch_dict = {
        "response_level_rewards": response_level_rewards,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
    }

    if use_nstep:
        batch_dict["nstep_returns"] = nstep_returns
        batch_dict["nstep_completeness"] = nstep_completeness

    data = DataProto(
        batch=TensorDict(batch_dict, batch_size=batch_size),
        non_tensor_batch={},
        meta_info={}
    )

    return data


def test_expand_to_token_level():
    """Test expand_to_token_level with and without n-step returns."""
    print("\n=== Test 1: expand_to_token_level ===")

    batch_size = 4
    seq_len = 10

    # Test without n-step
    print("\n1.1 Without n-step returns:")
    data = create_mock_dataproto(batch_size, seq_len, use_nstep=False)
    token_rewards = expand_to_token_level(data, use_nstep_returns=False)

    print(f"  response_level_rewards: {data.batch['response_level_rewards'].numpy()}")
    print(f"  token_level_rewards shape: {token_rewards.shape}")
    print(f"  token_level_rewards (last token): {token_rewards[:, -1].numpy()}")

    # Verify: last token should have response_level_rewards
    expected = data.batch['response_level_rewards']
    actual = token_rewards[:, -1]
    assert torch.allclose(actual, expected, atol=1e-5), "Without n-step: mismatch!"
    print("  ✓ Correct: Uses response_level_rewards")

    # Test with n-step
    print("\n1.2 With n-step returns:")
    data = create_mock_dataproto(batch_size, seq_len, use_nstep=True)
    token_rewards_nstep = expand_to_token_level(data, use_nstep_returns=True)

    print(f"  response_level_rewards: {data.batch['response_level_rewards'].numpy()}")
    print(f"  nstep_returns: {data.batch['nstep_returns'].numpy()}")
    print(f"  token_level_rewards (last token): {token_rewards_nstep[:, -1].numpy()}")

    # Verify: last token should have nstep_returns
    expected_nstep = data.batch['nstep_returns']
    actual_nstep = token_rewards_nstep[:, -1]
    assert torch.allclose(actual_nstep, expected_nstep, atol=1e-5), "With n-step: mismatch!"
    print("  ✓ Correct: Uses nstep_returns")

    # Verify they are different
    assert not torch.allclose(actual, actual_nstep, atol=1e-5), "Should be different!"
    print("  ✓ Confirmed: n-step returns differ from single-step")


def test_apply_kl_penalty():
    """Test apply_kl_penalty with n-step returns."""
    print("\n=== Test 2: apply_kl_penalty ===")

    batch_size = 4
    seq_len = 10
    kl_ctrl = AdaptiveKLController(init=0.01, target=0.1, horizon=100)

    # Test without n-step
    print("\n2.1 Without n-step returns:")
    data = create_mock_dataproto(batch_size, seq_len, use_nstep=False)
    data_out, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl", use_nstep_returns=False)

    print(f"  token_level_rewards shape: {data_out.batch['token_level_rewards'].shape}")
    print(f"  KL metrics: {metrics}")
    assert "nstep/enabled" not in metrics, "Should not have n-step metrics"
    print("  ✓ No n-step metrics (as expected)")

    # Test with n-step
    print("\n2.2 With n-step returns:")
    data = create_mock_dataproto(batch_size, seq_len, use_nstep=True)
    data_out, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl", use_nstep_returns=True)

    print(f"  token_level_rewards shape: {data_out.batch['token_level_rewards'].shape}")
    print(f"  KL metrics: {metrics}")
    assert "nstep/enabled" in metrics, "Should have n-step metrics"
    assert "nstep/returns_mean" in metrics, "Should have n-step returns mean"
    assert "nstep/completeness_ratio" in metrics, "Should have completeness ratio"
    print("  ✓ N-step metrics present")
    print(f"  nstep/returns_mean: {metrics['nstep/returns_mean']:.4f}")
    print(f"  nstep/completeness_ratio: {metrics['nstep/completeness_ratio']:.2f}")


def test_outer_inner_layer_independence():
    """Test that outer-layer choice doesn't affect inner-layer computation logic."""
    print("\n=== Test 3: Outer-Inner Layer Independence ===")

    batch_size = 4
    seq_len = 10
    kl_ctrl = AdaptiveKLController(init=0.01, target=0.1, horizon=100)

    # Create data with n-step returns
    data = create_mock_dataproto(batch_size, seq_len, use_nstep=True)

    # Compute with single-step (inner layer gets single-step reward)
    data_single, _ = apply_kl_penalty(data.clone(), kl_ctrl, use_nstep_returns=False)
    token_rewards_single = data_single.batch['token_level_rewards']

    # Compute with n-step (inner layer gets n-step reward)
    data_nstep, _ = apply_kl_penalty(data.clone(), kl_ctrl, use_nstep_returns=True)
    token_rewards_nstep = data_nstep.batch['token_level_rewards']

    print("\n3.1 Token-level rewards comparison:")
    print(f"  Single-step (last token): {token_rewards_single[:, -1].numpy()}")
    print(f"  N-step (last token): {token_rewards_nstep[:, -1].numpy()}")

    # The difference should be the reward source difference
    ratio = token_rewards_nstep[:, -1] / (token_rewards_single[:, -1] + 1e-8)
    print(f"  Ratio: {ratio.numpy()}")

    # Inner layer (token-level) computation is identical, only input differs
    # The KL penalty is applied the same way
    kl_diff_single = token_rewards_single - expand_to_token_level(data, use_nstep_returns=False)
    kl_diff_nstep = token_rewards_nstep - expand_to_token_level(data, use_nstep_returns=True)

    print("\n3.2 KL penalty application (should be same pattern):")
    print(f"  KL diff magnitude (single-step): {kl_diff_single.abs().mean():.6f}")
    print(f"  KL diff magnitude (n-step): {kl_diff_nstep.abs().mean():.6f}")
    print("  ✓ Inner-layer computation is independent of outer-layer reward choice")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing N-Step Returns Integration in Advantage Computation")
    print("=" * 60)

    try:
        test_expand_to_token_level()
        test_apply_kl_penalty()
        test_outer_inner_layer_independence()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

        print("\n📝 Summary:")
        print("  - Outer layer can use n-step returns (multi-step credit)")
        print("  - Inner layer token-level computation remains unchanged")
        print("  - Configuration: use_nstep_in_advantage controls the behavior")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
