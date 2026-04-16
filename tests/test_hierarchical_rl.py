"""
Unit tests for Hierarchical RL implementation.

Tests the core components:
- HierarchicalRLConfig validation
- StepLevelComputer
- TokenLevelComputer
- HierarchicalAdvantageComputer
"""

import pytest
import torch
import numpy as np

from roll.pipeline.agentic.hierarchical_config import (
    HierarchicalRLConfig,
    validate_hierarchical_config
)
from roll.pipeline.agentic.hierarchical_computer import (
    HierarchicalAdvantageComputer,
    StepLevelComputer,
    TokenLevelComputer
)


class TestHierarchicalConfig:
    """Test configuration and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HierarchicalRLConfig()
        assert config.enabled == False
        assert config.step_level_estimator == "gae"
        assert config.token_level_estimator == "gae"
        assert config.reward_assignment == "last_token"

    def test_config_validation_valid(self):
        """Test validation with valid config."""
        config = HierarchicalRLConfig(
            enabled=True,
            step_gamma=0.99,
            token_gamma=0.99,
            mixing_alpha=0.5
        )
        # Should not raise
        validate_hierarchical_config(config)

    def test_config_validation_invalid_alpha(self):
        """Test validation catches invalid mixing_alpha."""
        config = HierarchicalRLConfig(enabled=True, mixing_alpha=1.5)
        with pytest.raises(ValueError, match="mixing_alpha"):
            validate_hierarchical_config(config)

    def test_config_validation_invalid_gamma(self):
        """Test validation catches invalid gamma."""
        config = HierarchicalRLConfig(enabled=True, step_gamma=1.5)
        with pytest.raises(ValueError, match="step_gamma"):
            validate_hierarchical_config(config)


class TestStepLevelComputer:
    """Test step-level computations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = HierarchicalRLConfig(
            enabled=True,
            step_level_estimator="gae",
            step_gamma=0.99,
            step_lambda=0.95
        )
        self.computer = StepLevelComputer(self.config)

    def test_extract_last_token_value(self):
        """Test extracting step value from last token."""
        # Create mock token values: [batch=3, seq_len=5]
        token_values = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [2.0, 2.0, 2.0, 2.0, 2.0]
        ])

        # Masks indicate valid lengths: 5, 3, 2
        response_masks = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0]
        ])

        step_values = self.computer.extract_step_values(token_values, response_masks)

        # Should get last valid token for each sample
        assert step_values.shape == (3,)
        assert torch.isclose(step_values[0], torch.tensor(5.0))
        assert torch.isclose(step_values[1], torch.tensor(1.5))
        assert torch.isclose(step_values[2], torch.tensor(2.0))

    def test_extract_mean_value(self):
        """Test extracting step value via mean."""
        self.config.step_value_source = "mean_tokens"
        computer = StepLevelComputer(self.config)

        token_values = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 0.0, 0.0]
        ])
        response_masks = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0]
        ])

        step_values = computer.extract_step_values(token_values, response_masks)

        # Should get mean of valid tokens
        assert torch.isclose(step_values[0], torch.tensor(3.0))  # (1+2+3+4+5)/5
        assert torch.isclose(step_values[1], torch.tensor(2.0))  # (1+2+3)/3

    def test_compute_gae(self):
        """Test step-level GAE computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])

        advantages, returns = self.computer.compute_gae(rewards, values)

        # Check shapes
        assert advantages.shape == (3,)
        assert returns.shape == (3,)

        # Returns should be advantages + values
        assert torch.allclose(returns, advantages + values)

        # Last advantage should be roughly reward - value (terminal)
        # delta_2 = 3.0 + 0.99*0 - 1.5 = 1.5
        assert torch.isclose(advantages[2], torch.tensor(1.5), atol=0.01)

    def test_compute_nstep_returns(self):
        """Test n-step returns with bootstrap."""
        self.config.step_level_estimator = "nstep"
        self.config.step_n_steps = 2
        self.config.use_step_bootstrap = True
        computer = StepLevelComputer(self.config)

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([0.5, 1.0, 1.5, 2.0])

        advantages, returns = computer.compute_nstep_returns(rewards, values)

        # For step 0: R = 1.0 + 0.99*2.0 + 0.99^2*V(2) = 1.0 + 1.98 + 0.98*1.5
        expected_return_0 = 1.0 + 0.99 * 2.0 + (0.99 ** 2) * 1.5
        assert torch.isclose(returns[0], torch.tensor(expected_return_0), atol=0.01)


class TestTokenLevelComputer:
    """Test token-level computations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = HierarchicalRLConfig(
            enabled=True,
            reward_assignment="last_token",
            token_level_estimator="gae"
        )
        self.computer = TokenLevelComputer(self.config)

    def test_assign_last_token(self):
        """Test assigning reward to last token only."""
        step_returns = torch.tensor([10.0, 20.0])
        response_masks = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ])

        intrinsic_rewards = self.computer.assign_rewards_to_tokens(
            step_returns=step_returns,
            response_masks=response_masks
        )

        # Check shapes
        assert intrinsic_rewards.shape == (2, 4)

        # Only last valid token should have reward
        assert torch.isclose(intrinsic_rewards[0, 2], torch.tensor(10.0))
        assert torch.isclose(intrinsic_rewards[0, :2].sum(), torch.tensor(0.0))
        assert torch.isclose(intrinsic_rewards[1, 1], torch.tensor(20.0))

    def test_assign_uniform(self):
        """Test uniform reward assignment."""
        self.config.reward_assignment = "uniform"
        computer = TokenLevelComputer(self.config)

        step_returns = torch.tensor([6.0])
        response_masks = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        intrinsic_rewards = computer.assign_rewards_to_tokens(
            step_returns=step_returns,
            response_masks=response_masks
        )

        # Each valid token should get 6.0/3 = 2.0
        assert torch.allclose(intrinsic_rewards[0, :3], torch.tensor(2.0))

    def test_compute_token_gae(self):
        """Test token-level GAE computation."""
        batch_size, seq_len = 2, 4
        intrinsic_rewards = torch.tensor([
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 10.0, 0.0, 0.0]
        ])
        token_values = torch.tensor([
            [1.0, 2.0, 3.0, 0.0],
            [0.5, 1.0, 0.0, 0.0]
        ])
        response_masks = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ])

        advantages, returns = self.computer.compute_gae(
            rewards=intrinsic_rewards,
            values=token_values,
            mask=response_masks
        )

        # Check shapes
        assert advantages.shape == (2, 4)
        assert returns.shape == (2, 4)

        # Masked positions should be zero
        assert advantages[0, 3] == 0.0
        assert advantages[1, 2:].sum() == 0.0


class TestHierarchicalAdvantageComputer:
    """Test integrated hierarchical advantage computation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = HierarchicalRLConfig(
            enabled=True,
            step_level_estimator="gae",
            token_level_estimator="gae",
            reward_assignment="last_token",
            log_hierarchical_metrics=True
        )
        self.computer = HierarchicalAdvantageComputer(self.config)

    def test_end_to_end_computation(self):
        """Test full hierarchical computation pipeline."""
        batch_size, seq_len = 3, 5

        # Environment rewards (step-level)
        env_rewards = torch.tensor([10.0, 20.0, 15.0])

        # Token values from critic
        token_values = torch.randn(batch_size, seq_len)

        # Response masks
        response_masks = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0]
        ])

        # Compute hierarchical advantages
        results = self.computer.compute(
            env_rewards=env_rewards,
            token_values=token_values,
            response_masks=response_masks
        )

        # Check all expected keys are present
        assert "step_values" in results
        assert "step_advantages" in results
        assert "step_returns" in results
        assert "token_advantages" in results
        assert "token_returns" in results
        assert "metrics" in results

        # Check shapes
        assert results["step_values"].shape == (batch_size,)
        assert results["token_advantages"].shape == (batch_size, seq_len)

        # Check that masked positions are zero
        assert results["token_advantages"][1, 4] == 0.0
        assert results["token_advantages"][2, 3:].sum() == 0.0

        # Check metrics are computed
        assert "hierarchical/step_value/mean" in results["metrics"]
        assert "hierarchical/token_advantage/mean" in results["metrics"]

    def test_mixing_with_original_rewards(self):
        """Test mixing hierarchical with original rewards."""
        self.config.use_original_rewards = True
        self.config.mixing_alpha = 0.5
        computer = HierarchicalAdvantageComputer(self.config)

        batch_size, seq_len = 2, 3
        env_rewards = torch.tensor([5.0, 10.0])
        token_values = torch.ones(batch_size, seq_len)
        response_masks = torch.ones(batch_size, seq_len)
        original_token_rewards = torch.ones(batch_size, seq_len) * 2.0

        results = computer.compute(
            env_rewards=env_rewards,
            token_values=token_values,
            response_masks=response_masks,
            original_token_rewards=original_token_rewards
        )

        # Token advantages should be mixed
        # Should not be purely from hierarchical
        assert not torch.allclose(results["token_advantages"], original_token_rewards)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
