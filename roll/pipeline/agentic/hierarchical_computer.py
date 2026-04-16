"""
Hierarchical RL Advantage Computer

This module implements the core computation logic for hierarchical RL:
1. StepLevelComputer: Computes step-level values and advantages
2. TokenLevelComputer: Computes token-level advantages from intrinsic rewards
3. HierarchicalAdvantageComputer: Orchestrates the two-level computation
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

from .hierarchical_config import HierarchicalRLConfig

logger = logging.getLogger(__name__)


class StepLevelComputer:
    """
    Step-level advantage computation.

    Processes environment rewards to compute step-level values and advantages.
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config

    def extract_step_values(
        self,
        token_values: torch.Tensor,
        response_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract step-level values from token-level values.

        Args:
            token_values: [batch_size, seq_len] token-level values from critic
            response_masks: [batch_size, seq_len] mask for valid tokens

        Returns:
            step_values: [batch_size] step-level values
        """
        batch_size = token_values.size(0)
        step_values = torch.zeros(batch_size, device=token_values.device)

        if self.config.step_value_source == "last_token":
            # Use the last valid token's value as step value
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    step_values[i] = token_values[i, valid_len - 1]

        elif self.config.step_value_source == "mean_tokens":
            # Average over all valid tokens
            masked_values = token_values * response_masks
            valid_counts = response_masks.sum(dim=1).clamp(min=1)
            step_values = masked_values.sum(dim=1) / valid_counts

        elif self.config.step_value_source == "max_tokens":
            # Maximum value among valid tokens
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    step_values[i] = token_values[i, :valid_len].max()

        else:
            raise ValueError(f"Unknown step_value_source: {self.config.step_value_source}")

        if self.config.debug_mode:
            logger.debug(
                f"Extracted step values using {self.config.step_value_source}: "
                f"mean={step_values.mean():.4f}, std={step_values.std():.4f}"
            )

        return step_values

    def compute_step_advantages(
        self,
        env_rewards: torch.Tensor,
        step_values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute step-level advantages and returns.

        Args:
            env_rewards: [batch_size] environment rewards for each step
            step_values: [batch_size] step-level values
            dones: [batch_size] done flags (1.0 if episode ends, 0.0 otherwise)
                   Required for proper GAE/N-step computation with variable-length episodes

        Returns:
            advantages: [batch_size] step-level advantages
            returns: [batch_size] step-level returns
        """
        if self.config.step_level_estimator == "gae":
            return self.compute_gae(env_rewards, step_values, dones)
        elif self.config.step_level_estimator == "nstep":
            return self.compute_nstep_returns(env_rewards, step_values, dones)
        elif self.config.step_level_estimator == "monte_carlo":
            return self.compute_monte_carlo(env_rewards, step_values, dones)
        else:
            raise ValueError(f"Unknown step_level_estimator: {self.config.step_level_estimator}")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation at step-level.

        Following Stable-Baselines3/Tianshou convention:
        - Uses done mask to properly handle episode boundaries
        - next_non_terminal = 1.0 - done[t] zeroes out bootstrap at episode end
        - Supports variable-length episodes naturally

        Args:
            rewards: [batch_size] step rewards
            values: [batch_size] step values
            dones: [batch_size] done flags (1.0 if episode ends at this step, 0.0 otherwise)
                   If None, assumes last step is terminal (legacy behavior)

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        gamma = self.config.step_gamma
        lambda_ = self.config.step_lambda

        # If no dones provided, create a default (only last step is terminal)
        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1.0

        # Compute GAE from last to first
        # Following Stable-Baselines3 convention:
        # next_non_terminal = 1.0 - done[t]
        # This zeroes out the bootstrap value at episode boundaries
        for t in reversed(range(batch_size)):
            if t < batch_size - 1:
                nextvalue = values[t + 1]
                # At episode end (done[t]=1), next_non_terminal=0, so bootstrap is zeroed
                next_non_terminal = 1.0 - dones[t]
            else:
                nextvalue = 0.0
                next_non_terminal = 0.0  # Last step in batch, no bootstrap

            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = rewards[t] + gamma * nextvalue * next_non_terminal - values[t]

            # GAE accumulation: A_t = δ_t + γ * λ * A_{t+1} * (1 - done_t)
            # When done[t]=1, GAE resets (lastgaelam contribution is zeroed)
            lastgaelam = delta + gamma * lambda_ * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        if self.config.debug_mode:
            num_episodes = int(dones.sum().item()) if dones is not None else 1
            logger.debug(
                f"Step-level GAE computed: "
                f"adv_mean={advantages.mean():.4f}, "
                f"ret_mean={returns.mean():.4f}, "
                f"num_episode_ends={num_episodes}"
            )

        return advantages, returns

    def compute_nstep_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute N-step returns with bootstrap at step-level.

        Following Stable-Baselines3/Tianshou convention:
        - Uses done mask to properly handle episode boundaries
        - Stops accumulation when hitting a done flag
        - Supports variable-length episodes naturally

        Args:
            rewards: [batch_size] step rewards
            values: [batch_size] step values
            dones: [batch_size] done flags (1.0 if episode ends, 0.0 otherwise)
                   If None, assumes last step is terminal

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        batch_size = rewards.size(0)
        returns = torch.zeros_like(rewards)

        gamma = self.config.step_gamma
        n_steps = self.config.step_n_steps

        # If no dones provided, create a default (only last step is terminal)
        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1.0

        for t in range(batch_size):
            # Compute n-step return starting from t
            n_step_return = 0.0
            discount = 1.0
            hit_terminal = False

            # Accumulate discounted rewards, stopping at episode boundaries
            for k in range(min(n_steps, batch_size - t)):
                n_step_return += discount * rewards[t + k]

                # Check if this step is terminal (episode ends here)
                if dones[t + k] > 0.5:  # done flag is set
                    hit_terminal = True
                    break

                discount *= gamma

            # Add bootstrap value if:
            # 1. We haven't hit a terminal state
            # 2. We have steps remaining for bootstrap
            # 3. Bootstrap is enabled
            if (not hit_terminal and
                self.config.use_step_bootstrap and
                t + n_steps < batch_size):
                bootstrap_value = values[t + n_steps]
                n_step_return += discount * gamma * bootstrap_value

            returns[t] = n_step_return

        advantages = returns - values

        if self.config.debug_mode:
            num_episodes = int(dones.sum().item()) if dones is not None else 1
            logger.debug(
                f"Step-level n-step returns computed: "
                f"n_steps={n_steps}, "
                f"ret_mean={returns.mean():.4f}, "
                f"num_episode_ends={num_episodes}"
            )

        return advantages, returns

    def compute_monte_carlo(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Monte Carlo returns at step-level.

        Following Stable-Baselines3/Tianshou convention:
        - Uses done mask to reset return accumulation at episode boundaries
        - Supports variable-length episodes naturally

        Args:
            rewards: [batch_size] step rewards
            values: [batch_size] step values
            dones: [batch_size] done flags (1.0 if episode ends, 0.0 otherwise)
                   If None, assumes last step is terminal

        Returns:
            advantages: [batch_size]
            returns: [batch_size]
        """
        batch_size = rewards.size(0)
        returns = torch.zeros_like(rewards)

        gamma = self.config.step_gamma

        # If no dones provided, create a default (only last step is terminal)
        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1.0

        # Compute cumulative returns from last to first
        # Reset cumulative_return when we hit an episode boundary (done=1)
        cumulative_return = 0.0
        for t in reversed(range(batch_size)):
            # At episode end, next step's return shouldn't contribute
            # next_non_terminal zeroes the future contribution
            next_non_terminal = 1.0 - dones[t]
            cumulative_return = rewards[t] + gamma * cumulative_return * next_non_terminal
            returns[t] = cumulative_return

        advantages = returns - values

        if self.config.debug_mode:
            num_episodes = int(dones.sum().item()) if dones is not None else 1
            logger.debug(
                f"Step-level Monte Carlo returns computed: "
                f"ret_mean={returns.mean():.4f}, "
                f"num_episode_ends={num_episodes}"
            )

        return advantages, returns


class TokenLevelComputer:
    """
    Token-level advantage computation.

    Receives intrinsic rewards from step-level and computes token-level advantages.
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config

    def assign_rewards_to_tokens(
        self,
        step_returns: torch.Tensor,
        response_masks: torch.Tensor,
        token_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Assign step-level returns to tokens as intrinsic rewards.

        This is the key connection between upper and lower levels!

        Args:
            step_returns: [batch_size] step-level returns from upper level
            response_masks: [batch_size, seq_len] valid token masks
            token_values: [batch_size, seq_len] optional, for value_weighted assignment

        Returns:
            token_intrinsic_rewards: [batch_size, seq_len]
        """
        batch_size, seq_len = response_masks.shape
        intrinsic_rewards = torch.zeros(batch_size, seq_len, device=step_returns.device)

        if self.config.reward_assignment == "last_token":
            # Only last token gets the reward
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    intrinsic_rewards[i, valid_len - 1] = step_returns[i]

        elif self.config.reward_assignment == "uniform":
            # Uniform distribution across tokens
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    intrinsic_rewards[i, :valid_len] = step_returns[i] / valid_len

        elif self.config.reward_assignment == "exponential":
            # Exponentially decaying from last to first
            gamma = self.config.token_gamma
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    # Compute weights: [gamma^(n-1), gamma^(n-2), ..., gamma^0]
                    weights = torch.pow(gamma, torch.arange(valid_len - 1, -1, -1, device=step_returns.device))
                    weights = weights / weights.sum()  # Normalize
                    intrinsic_rewards[i, :valid_len] = step_returns[i] * weights

        elif self.config.reward_assignment == "value_weighted":
            # Weight by token value contributions
            if token_values is None:
                raise ValueError("value_weighted assignment requires token_values")

            temp = self.config.assignment_temperature
            for i in range(batch_size):
                valid_len = int(response_masks[i].sum())
                if valid_len > 0:
                    valid_values = token_values[i, :valid_len]
                    weights = torch.softmax(valid_values / temp, dim=0)
                    intrinsic_rewards[i, :valid_len] = step_returns[i] * weights

        else:
            raise ValueError(f"Unknown reward_assignment: {self.config.reward_assignment}")

        if self.config.debug_mode:
            logger.debug(
                f"Assigned rewards using {self.config.reward_assignment}: "
                f"mean={intrinsic_rewards.mean():.4f}, "
                f"nonzero_ratio={(intrinsic_rewards != 0).float().mean():.4f}"
            )

        return intrinsic_rewards

    def compute_token_advantages(
        self,
        intrinsic_rewards: torch.Tensor,
        token_values: torch.Tensor,
        response_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level advantages using intrinsic rewards.

        Args:
            intrinsic_rewards: [batch_size, seq_len] from upper level
            token_values: [batch_size, seq_len] from critic
            response_masks: [batch_size, seq_len] valid token masks

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        if self.config.token_level_estimator == "gae":
            return self.compute_gae(intrinsic_rewards, token_values, response_masks)
        elif self.config.token_level_estimator == "reinforce":
            return self.compute_reinforce(intrinsic_rewards, response_masks)
        elif self.config.token_level_estimator == "reinforce_baseline":
            return self.compute_reinforce_baseline(intrinsic_rewards, token_values, response_masks)
        elif self.config.token_level_estimator == "direct":
            return self.compute_direct(intrinsic_rewards, token_values, response_masks)
        else:
            raise ValueError(f"Unknown token_level_estimator: {self.config.token_level_estimator}")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE at token-level within each step.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            values: [batch_size, seq_len] token values
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)

        gamma = self.config.token_gamma
        lambda_ = self.config.token_lambda

        # Compute GAE for each sample independently
        for i in range(batch_size):
            valid_len = int(mask[i].sum())
            if valid_len == 0:
                continue

            lastgaelam = 0
            for t in reversed(range(valid_len)):
                if t < valid_len - 1:
                    nextvalue = values[i, t + 1]
                else:
                    nextvalue = 0.0  # Terminal

                # TD error
                delta = rewards[i, t] + gamma * nextvalue - values[i, t]

                # GAE accumulation
                lastgaelam = delta + gamma * lambda_ * lastgaelam
                advantages[i, t] = lastgaelam

        # Apply mask
        advantages = advantages * mask
        returns = advantages + values * mask

        return advantages, returns

    def compute_reinforce(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute REINFORCE (Monte Carlo) returns.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)

        gamma = self.config.token_gamma

        # Compute Monte Carlo returns for each sample
        for i in range(batch_size):
            valid_len = int(mask[i].sum())
            if valid_len == 0:
                continue

            cumulative_return = 0.0
            for t in reversed(range(valid_len)):
                cumulative_return = rewards[i, t] + gamma * cumulative_return
                returns[i, t] = cumulative_return

        # No baseline, advantages = returns
        advantages = returns * mask

        return advantages, returns

    def compute_reinforce_baseline(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute REINFORCE with value baseline.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            values: [batch_size, seq_len] token values (baseline)
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        # First compute MC returns
        _, returns = self.compute_reinforce(rewards, mask)

        # Subtract baseline
        advantages = (returns - values) * mask

        return advantages, returns

    def compute_direct(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Directly use intrinsic rewards as advantages.

        Args:
            rewards: [batch_size, seq_len] intrinsic rewards
            values: [batch_size, seq_len] token values
            mask: [batch_size, seq_len] valid token mask

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        advantages = rewards * mask
        returns = (rewards + values) * mask

        return advantages, returns


class HierarchicalAdvantageComputer:
    """
    Main orchestrator for hierarchical RL advantage computation.

    Coordinates step-level and token-level computations.
    """

    def __init__(self, config: HierarchicalRLConfig):
        self.config = config
        self.step_computer = StepLevelComputer(config)
        self.token_computer = TokenLevelComputer(config)

    def compute(
        self,
        env_rewards: torch.Tensor,
        token_values: torch.Tensor,
        response_masks: torch.Tensor,
        original_token_rewards: Optional[torch.Tensor] = None,
        episode_boundaries: Optional[List[int]] = None,
        steps_per_episode: Optional[int] = None,
        dones: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main entry point: Compute hierarchical advantages.

        Following Stable-Baselines3/Tianshou convention:
        - Uses done mask to properly handle episode boundaries
        - Supports variable-length episodes naturally
        - Falls back to episode_boundaries for backward compatibility

        Args:
            env_rewards: [batch_size] environment rewards for each step
            token_values: [batch_size, seq_len] critic values for tokens
            response_masks: [batch_size, seq_len] valid token masks
            original_token_rewards: [batch_size, seq_len] optional, for mixing
            episode_boundaries: List of starting indices for each episode in the batch.
                              (Deprecated: prefer using dones tensor)
                              Example: [0, 4, 8, 12] means:
                              - Episode 0: indices [0,1,2,3]
                              - Episode 1: indices [4,5,6,7]
                              - Episode 2: indices [8,9,10,11]
            steps_per_episode: Number of consecutive steps per episode (deprecated, use dones)
            dones: [batch_size] done flags (1.0 if episode ends, 0.0 otherwise)
                   PREFERRED: This is the standard way to handle episode boundaries
                   following Stable-Baselines3/Tianshou convention.

        Returns:
            Dictionary containing:
            - step_values: [batch_size]
            - step_advantages: [batch_size]
            - step_returns: [batch_size]
            - token_advantages: [batch_size, seq_len]
            - token_returns: [batch_size, seq_len]
            - metrics: dict of metrics
        """
        metrics = {}
        batch_size = env_rewards.size(0)

        # Validate episode structure for step-level estimators that need temporal continuity
        requires_episode_structure = self.config.step_level_estimator in ["gae", "nstep"]

        # Prefer dones tensor over episode_boundaries
        has_episode_info = dones is not None or episode_boundaries is not None

        if requires_episode_structure and not has_episode_info:
            logger.error(
                f"Episode structure is REQUIRED for step_level_estimator='{self.config.step_level_estimator}' "
                f"to compute correct bootstrap values. "
                f"Got dones={dones is not None}, episode_boundaries={episode_boundaries is not None}. "
                f"This will cause training instability and IS ratio explosion!"
            )
            raise ValueError(
                f"Hierarchical RL with {self.config.step_level_estimator} requires either dones tensor or episode_boundaries. "
                f"Ensure done signal is present in batch (for fresh batch) or use "
                f"replay_buffer.sample_episodes_for_hierarchical() (for replay batch)."
            )

        # Step 1: Extract step-level values from token values
        step_values = self.step_computer.extract_step_values(
            token_values=token_values,
            response_masks=response_masks
        )

        # Step 2: Compute step-level advantages and returns
        # PREFERRED: Use dones tensor directly (Stable-Baselines3/Tianshou style)
        # FALLBACK: Use episode_boundaries (legacy, for backward compatibility)
        if dones is not None:
            # Modern approach: Use dones tensor directly
            # This handles variable-length episodes naturally
            step_advantages, step_returns = self.step_computer.compute_step_advantages(
                env_rewards=env_rewards,
                step_values=step_values,
                dones=dones
            )

            num_episodes = int(dones.sum().item())
            logger.debug(
                f"Computed step-level advantages using dones tensor: "
                f"{num_episodes} episode ends in batch of {batch_size} steps, "
                f"estimator={self.config.step_level_estimator}"
            )
            metrics["hierarchical/num_episodes"] = num_episodes

        elif episode_boundaries is not None:
            # Legacy approach: Use episode_boundaries
            # This requires fixed-length episodes
            logger.warning(
                f"Using legacy episode_boundaries instead of dones tensor. "
                f"Consider using dones for better variable-length episode support."
            )
            num_episodes = len(episode_boundaries)
            step_advantages = torch.zeros_like(env_rewards)
            step_returns = torch.zeros_like(env_rewards)

            for ep_idx in range(num_episodes):
                start_idx = episode_boundaries[ep_idx]
                # Determine end_idx: either next episode's start or end of batch
                if ep_idx + 1 < num_episodes:
                    end_idx = episode_boundaries[ep_idx + 1]
                else:
                    end_idx = batch_size

                # Extract this episode's data
                ep_env_rewards = env_rewards[start_idx:end_idx]
                ep_step_values = step_values[start_idx:end_idx]

                # Create dones for this episode (only last step is done)
                ep_dones = torch.zeros_like(ep_env_rewards)
                ep_dones[-1] = 1.0

                # Compute advantages for this episode independently
                ep_advantages, ep_returns = self.step_computer.compute_step_advantages(
                    env_rewards=ep_env_rewards,
                    step_values=ep_step_values,
                    dones=ep_dones
                )

                # Place results back into full batch
                step_advantages[start_idx:end_idx] = ep_advantages
                step_returns[start_idx:end_idx] = ep_returns

            actual_steps_per_ep = steps_per_episode if steps_per_episode else "variable"
            logger.debug(
                f"Computed step-level advantages for {num_episodes} episodes "
                f"({actual_steps_per_ep} steps/episode) using {self.config.step_level_estimator}"
            )
            metrics["hierarchical/num_episodes"] = num_episodes
            metrics["hierarchical/steps_per_episode"] = steps_per_episode

        else:
            # No episode info: Compute over entire batch (only OK for monte_carlo)
            logger.warning(
                f"Computing step-level advantages without episode structure! "
                f"This is INCORRECT for {self.config.step_level_estimator} and will cause training issues."
            )
            step_advantages, step_returns = self.step_computer.compute_step_advantages(
                env_rewards=env_rewards,
                step_values=step_values,
                dones=None  # Will create default (only last step is terminal)
            )

        # Step 3: Assign step returns to tokens as intrinsic rewards
        intrinsic_rewards = self.token_computer.assign_rewards_to_tokens(
            step_returns=step_returns,
            response_masks=response_masks,
            token_values=token_values if self.config.reward_assignment == "value_weighted" else None
        )

        # Step 4: Compute token-level advantages
        token_advantages, token_returns = self.token_computer.compute_token_advantages(
            intrinsic_rewards=intrinsic_rewards,
            token_values=token_values,
            response_masks=response_masks
        )

        # Step 5: Optional mixing with original rewards
        if self.config.use_original_rewards and original_token_rewards is not None:
            alpha = self.config.mixing_alpha
            token_advantages = alpha * token_advantages + (1 - alpha) * original_token_rewards
            metrics["hierarchical/mixing_alpha"] = alpha

        # Step 6: Collect metrics
        if self.config.log_hierarchical_metrics:
            metrics.update(self._compute_metrics(
                step_values=step_values,
                step_advantages=step_advantages,
                step_returns=step_returns,
                token_advantages=token_advantages,
                token_returns=token_returns,
                intrinsic_rewards=intrinsic_rewards,
                env_rewards=env_rewards
            ))
            # Note: episode metrics are already added in the dones/episode_boundaries branches above

        return {
            "step_values": step_values,
            "step_advantages": step_advantages,
            "step_returns": step_returns,
            "token_advantages": token_advantages,
            "token_returns": token_returns,
            "intrinsic_rewards": intrinsic_rewards,
            "metrics": metrics
        }

    def _compute_metrics(
        self,
        step_values: torch.Tensor,
        step_advantages: torch.Tensor,
        step_returns: torch.Tensor,
        token_advantages: torch.Tensor,
        token_returns: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        env_rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Compute detailed metrics for monitoring."""
        metrics = {}

        # Step-level metrics
        metrics["hierarchical/step_value/mean"] = step_values.mean().item()
        metrics["hierarchical/step_value/std"] = step_values.std().item()
        metrics["hierarchical/step_value/max"] = step_values.max().item()
        metrics["hierarchical/step_value/min"] = step_values.min().item()

        metrics["hierarchical/step_advantage/mean"] = step_advantages.mean().item()
        metrics["hierarchical/step_advantage/std"] = step_advantages.std().item()

        metrics["hierarchical/step_return/mean"] = step_returns.mean().item()
        metrics["hierarchical/step_return/std"] = step_returns.std().item()

        # Token-level metrics
        metrics["hierarchical/token_advantage/mean"] = token_advantages.mean().item()
        metrics["hierarchical/token_advantage/std"] = token_advantages.std().item()

        metrics["hierarchical/token_return/mean"] = token_returns.mean().item()
        metrics["hierarchical/token_return/std"] = token_returns.std().item()

        # Intrinsic reward metrics
        metrics["hierarchical/intrinsic_reward/mean"] = intrinsic_rewards.mean().item()
        metrics["hierarchical/intrinsic_reward/std"] = intrinsic_rewards.std().item()
        metrics["hierarchical/intrinsic_reward/nonzero_ratio"] = (intrinsic_rewards != 0).float().mean().item()

        # Correlation between step and env rewards
        if len(step_returns) > 1:
            corr = torch.corrcoef(torch.stack([step_returns, env_rewards]))[0, 1]
            metrics["hierarchical/step_env_correlation"] = corr.item() if not torch.isnan(corr) else 0.0

        return metrics
