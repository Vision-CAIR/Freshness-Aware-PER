"""
Test script for V-trace implementation in ROLL framework.
"""

import torch
import numpy as np
from roll.utils.functionals import compute_vtrace_advantage_return, compute_gae_advantage_return
from roll.distributed.scheduler.protocol import DataProto


def test_vtrace_basic():
    """Test basic V-trace computation."""
    print("Testing basic V-trace computation...")

    # Create sample data
    batch_size = 4
    seq_len = 10

    # Generate random rewards and values
    token_level_rewards = torch.randn(batch_size, seq_len) * 0.1
    values = torch.randn(batch_size, seq_len) * 0.5

    # Generate log probabilities (seq_len - 1)
    old_log_probs = torch.randn(batch_size, seq_len - 1) * 0.5 - 2.0  # Behavior policy
    current_log_probs = old_log_probs + torch.randn(batch_size, seq_len - 1) * 0.2  # Target policy (slightly different)

    # Create response mask
    response_mask = torch.ones(batch_size, seq_len - 1)
    # Mask out some tokens to simulate variable length sequences
    response_mask[0, 7:] = 0
    response_mask[1, 8:] = 0

    # Compute V-trace advantages and returns
    advantages, returns = compute_vtrace_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        old_log_probs=old_log_probs,
        current_log_probs=current_log_probs,
        response_mask=response_mask,
        gamma=0.99,
        rho_bar=1.0,
        c_bar=1.0,
    )

    # Check shapes
    assert advantages.shape == (batch_size, seq_len), f"Advantages shape mismatch: {advantages.shape}"
    assert returns.shape == (batch_size, seq_len), f"Returns shape mismatch: {returns.shape}"

    # Check masking is applied
    assert torch.all(advantages[0, 8:] == 0), "Advantages not properly masked"
    assert torch.all(returns[0, 8:] == 0), "Returns not properly masked"

    print(f"✓ Basic V-trace test passed")
    print(f"  - Advantages shape: {advantages.shape}")
    print(f"  - Returns shape: {returns.shape}")
    print(f"  - Mean advantage: {advantages.mean().item():.4f}")
    print(f"  - Mean return: {returns.mean().item():.4f}")


def test_vtrace_vs_gae():
    """Compare V-trace with GAE when policies are identical (on-policy case)."""
    print("\nTesting V-trace vs GAE (on-policy case)...")

    batch_size = 2
    seq_len = 8

    # Generate data
    token_level_rewards = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
                                        [0.2, 0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float32)
    values = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
                          [0.4, 0.3, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float32)

    # For on-policy case, old and current log probs are the same
    log_probs = torch.randn(batch_size, seq_len - 1) * 0.5 - 2.0

    response_mask = torch.ones(batch_size, seq_len - 1)
    response_mask[0, 5:] = 0
    response_mask[1, 5:] = 0

    gamma = 0.99
    lambd = 0.95

    # Compute V-trace (with identical policies, should be similar to GAE)
    vtrace_adv, vtrace_ret = compute_vtrace_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        old_log_probs=log_probs,
        current_log_probs=log_probs,  # Same as old for on-policy
        response_mask=response_mask,
        gamma=gamma,
        rho_bar=float('inf'),  # No truncation for on-policy
        c_bar=float('inf'),
    )

    # Compute GAE for comparison
    gae_adv, gae_ret = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        gamma=torch.tensor(gamma),
        lambd=torch.tensor(lambd),
    )

    # Apply mask to GAE results for fair comparison
    gae_adv = gae_adv * torch.nn.functional.pad(response_mask, (0, 1), value=0.0)
    gae_ret = gae_ret * torch.nn.functional.pad(response_mask, (0, 1), value=0.0)

    print(f"✓ V-trace vs GAE comparison completed")
    print(f"  - V-trace mean advantage: {vtrace_adv.mean().item():.4f}")
    print(f"  - GAE mean advantage: {gae_adv.mean().item():.4f}")
    print(f"  - Difference in advantages: {(vtrace_adv - gae_adv).abs().mean().item():.4f}")


def test_vtrace_importance_sampling():
    """Test V-trace with different importance sampling ratios."""
    print("\nTesting V-trace importance sampling...")

    batch_size = 2
    seq_len = 6

    # Simple rewards and values
    token_level_rewards = torch.ones(batch_size, seq_len) * 0.1
    values = torch.zeros(batch_size, seq_len)

    # Create significant policy difference
    old_log_probs = torch.full((batch_size, seq_len - 1), -2.0)

    # Test with different policy divergences
    divergences = [0.0, 0.5, 1.0, 2.0]

    for divergence in divergences:
        current_log_probs = old_log_probs + divergence

        response_mask = torch.ones(batch_size, seq_len - 1)

        advantages, returns = compute_vtrace_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            old_log_probs=old_log_probs,
            current_log_probs=current_log_probs,
            response_mask=response_mask,
            gamma=0.99,
            rho_bar=1.0,  # Truncate at 1.0
            c_bar=1.0,
        )

        # Compute actual importance sampling ratio
        is_ratio = torch.exp(current_log_probs - old_log_probs).mean().item()
        truncated_ratio = min(1.0, is_ratio)

        print(f"  - Divergence: {divergence:.1f}, IS ratio: {is_ratio:.3f}, "
              f"Truncated: {truncated_ratio:.3f}, Mean advantage: {advantages.mean().item():.4f}")

    print(f"✓ Importance sampling test completed")


def test_vtrace_truncation():
    """Test effect of different truncation thresholds."""
    print("\nTesting V-trace truncation thresholds...")

    batch_size = 2
    seq_len = 8

    # Generate data
    token_level_rewards = torch.randn(batch_size, seq_len) * 0.1
    values = torch.randn(batch_size, seq_len) * 0.2

    # Create policy divergence
    old_log_probs = torch.randn(batch_size, seq_len - 1) * 0.5 - 2.0
    current_log_probs = old_log_probs + torch.randn(batch_size, seq_len - 1) * 1.0

    response_mask = torch.ones(batch_size, seq_len - 1)

    # Test different truncation thresholds
    thresholds = [0.5, 1.0, 2.0, float('inf')]

    for threshold in thresholds:
        advantages, returns = compute_vtrace_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            old_log_probs=old_log_probs,
            current_log_probs=current_log_probs,
            response_mask=response_mask,
            gamma=0.99,
            rho_bar=threshold,
            c_bar=threshold,
        )

        threshold_str = "∞" if threshold == float('inf') else f"{threshold:.1f}"
        print(f"  - Threshold (ρ̄=c̄={threshold_str}): "
              f"Adv mean={advantages.mean().item():.4f}, "
              f"Adv std={advantages.std().item():.4f}, "
              f"Ret mean={returns.mean().item():.4f}")

    print(f"✓ Truncation threshold test completed")


def test_vtrace_with_dataproto():
    """Test V-trace integration with DataProto."""
    print("\nTesting V-trace with DataProto integration...")

    from roll.utils.functionals import compute_advantage

    batch_size = 4
    seq_len = 10

    # Create DataProto with necessary fields
    batch_dict = {
        "token_level_rewards": torch.randn(batch_size, seq_len) * 0.1,
        "values": torch.randn(batch_size, seq_len) * 0.5,
        "old_log_probs": torch.randn(batch_size, seq_len - 1) * 0.5 - 2.0,
        "log_probs": torch.randn(batch_size, seq_len - 1) * 0.5 - 1.8,  # Current policy
        "response_mask": torch.ones(batch_size, seq_len),
    }

    # Mask some sequences
    batch_dict["response_mask"][0, 7:] = 0
    batch_dict["response_mask"][1, 8:] = 0

    # Create DataProto
    from tensordict import TensorDict
    tensor_batch = TensorDict(batch_dict, batch_size=[batch_size])
    data = DataProto(batch=tensor_batch, non_tensor_batch={})

    # Add V-trace parameters to meta_info
    data.meta_info = {
        "vtrace_rho_bar": 1.5,
        "vtrace_c_bar": 1.0,
    }

    # Compute advantages using V-trace
    compute_advantage(
        data=data,
        gamma=0.99,
        lambd=0.95,  # Not used in V-trace
        adv_estimator="vtrace",
        advantage_clip=None,
        whiten_advantages=False,
        whiten_rewards=False,
    )

    # Check that advantages and returns were computed
    assert "advantages" in data.batch, "Advantages not computed"
    assert "returns" in data.batch, "Returns not computed"
    assert "raw_advantages" in data.batch, "Raw advantages not stored"

    print(f"✓ DataProto integration test passed")
    print(f"  - Advantages shape: {data.batch['advantages'].shape}")
    print(f"  - Returns shape: {data.batch['returns'].shape}")
    print(f"  - Mean advantage: {data.batch['advantages'].mean().item():.4f}")
    print(f"  - Used ρ̄={data.meta_info['vtrace_rho_bar']}, c̄={data.meta_info['vtrace_c_bar']}")


if __name__ == "__main__":
    print("=" * 60)
    print("V-trace Implementation Tests for ROLL Framework")
    print("=" * 60)

    # Run all tests
    test_vtrace_basic()
    test_vtrace_vs_gae()
    test_vtrace_importance_sampling()
    test_vtrace_truncation()
    test_vtrace_with_dataproto()

    print("\n" + "=" * 60)
    print("✓ All V-trace tests passed successfully!")
    print("=" * 60)