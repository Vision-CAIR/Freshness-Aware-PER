"""
Unified Off-Policy Monitoring for ROLL Framework

This module provides centralized off-policy monitoring for replay buffer training,
ensuring consistent metrics calculation across different replay buffer implementations.
"""

import torch
from typing import Dict, Optional, Any
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


def compute_offpolicy_metrics(
    current_batch: DataProto,
    actor_train_cluster: Any = None,
    pg_clip: Optional[float] = None,
    training_metrics: Optional[DataProto] = None,
) -> Dict[str, float]:
    """
    Compute off-policy metrics by comparing current policy with behavior policy.

    This function provides a unified way to calculate off-policy metrics for any
    batch sampled from replay buffer, using trajectory mode (ROLL original design).

    所有指标统一使用 'offpolicy/' 前缀，便于监控和调试。

    OPTIMIZATION: If training_metrics is provided, it will reuse the log_probs computed
    during training, avoiding redundant forward pass. This is the recommended usage.

    Args:
        current_batch: DataProto batch containing data from replay buffer
        actor_train_cluster: Actor cluster for computing current policy log probs (optional if training_metrics provided)
        pg_clip: Clipping threshold for PPO-style ratio clipping analysis
        training_metrics: Optional DataProto from train_step containing pre-computed log_probs

    Returns:
        Dictionary of off-policy metrics (all with 'offpolicy/' prefix)
    """
    metrics = {}

    try:
        # Ensure batch has required fields
        if current_batch is None or current_batch.batch is None:
            logger.warning("compute_offpolicy_metrics: current_batch is None or empty")
            return metrics

        # Check for behavior log probs (stored when data was generated)
        # For fresh batch, use old_log_probs (PPO's actual old policy)
        # For replay batch, use behavior_log_probs (stored at generation time)
        behavior_field = None
        if "old_log_probs" in current_batch.batch:
            behavior_field = "old_log_probs"  # Priority: PPO's old_log_probs for fresh batch
        elif "behavior_log_probs" in current_batch.batch:
            behavior_field = "behavior_log_probs"  # Fallback: replay buffer's stored log_probs
        else:
            logger.warning("compute_offpolicy_metrics: No behavior log probs found in batch")
            return metrics

        # Check for response mask
        if "response_mask" not in current_batch.batch:
            logger.warning("compute_offpolicy_metrics: No response_mask in batch")
            return metrics

        # ✨ OPTIMIZATION: Try to reuse log_probs from training_metrics
        current_log_probs = None
        if training_metrics is not None and training_metrics.batch is not None:
            if "log_probs" in training_metrics.batch:
                current_log_probs = training_metrics.batch["log_probs"]
                logger.debug("compute_offpolicy_metrics: Reusing log_probs from training_metrics (no extra forward)")
            elif "policy_chosen_logps" in training_metrics.batch:
                # Alternative field name (some implementations use this)
                current_log_probs = training_metrics.batch["policy_chosen_logps"]
                logger.debug("compute_offpolicy_metrics: Reusing policy_chosen_logps from training_metrics")

        # Fallback: compute current policy log probs if not provided
        if current_log_probs is None:
            if actor_train_cluster is None:
                logger.warning("compute_offpolicy_metrics: No training_metrics and no actor_train_cluster provided")
                return metrics

            logger.debug("compute_offpolicy_metrics: Computing current log_probs via forward pass (fallback)")

            import ray
            current_lp_refs = actor_train_cluster.compute_log_probs(current_batch, blocking=False)
            current_lp_data = DataProto.materialize_concat(data_refs=current_lp_refs)

            if "log_probs" not in current_lp_data.batch:
                logger.warning("compute_offpolicy_metrics: Failed to compute current log_probs")
                return metrics

            current_log_probs = current_lp_data.batch["log_probs"]

        # Extract behavior log probs
        behavior_log_probs = current_batch.batch[behavior_field]

        # Handle next-token prediction alignment (response_mask is shifted by 1)
        response_mask = current_batch.batch["response_mask"][:, 1:].bool()

        # Ensure shapes match by truncating to minimum length
        min_seq_len = min(
            current_log_probs.shape[1],
            behavior_log_probs.shape[1],
            response_mask.shape[1]
        )

        current_log_probs = current_log_probs[:, :min_seq_len]
        behavior_log_probs = behavior_log_probs[:, :min_seq_len]
        response_mask = response_mask[:, :min_seq_len]

        # Apply mask to get valid positions only
        valid_current = current_log_probs[response_mask]
        valid_behavior = behavior_log_probs[response_mask]

        if valid_current.numel() == 0 or valid_behavior.numel() == 0:
            logger.warning(f"compute_offpolicy_metrics: No valid tokens after masking (current: {valid_current.numel()}, behavior: {valid_behavior.numel()})")
            return metrics

        # Compute off-policy statistics
        log_ratio = valid_current - valid_behavior
        ratio = log_ratio.exp()  # This is the importance weight

        # ========== Core Distribution Statistics ==========
        # Token-level importance weight statistics (for alignment with training)
        metrics["offpolicy/importance_weight/mean"] = ratio.mean().detach().item()
        metrics["offpolicy/importance_weight/std"] = ratio.std().detach().item()
        metrics["offpolicy/importance_weight/median"] = ratio.median().detach().item()
        metrics["offpolicy/importance_weight/max"] = ratio.max().detach().item()
        metrics["offpolicy/importance_weight/min"] = ratio.min().detach().item()

        # Sample-level statistics (for alignment with filtering and ROLL's seq mode)
        # Use geometric mean: exp(mean(log_ratio)) instead of mean(exp(log_ratio))
        # First reshape log_ratio back to [batch_size, seq_len] from flattened valid tokens
        batch_size = response_mask.shape[0]
        seq_len = response_mask.shape[1]

        # Create a full log_ratio tensor and fill in the valid positions
        full_log_ratio = torch.zeros(batch_size, seq_len, device=log_ratio.device, dtype=log_ratio.dtype)
        full_log_ratio[response_mask] = log_ratio

        # Compute per-sample ratio using geometric mean (consistent with ROLL's seq mode and filter)
        # 1. First compute mean of log_ratio (in log space)
        # 2. Then exp to get the ratio
        valid_tokens_per_sample = response_mask.sum(dim=1).clamp(min=1)  # [batch_size]
        masked_log_ratio = (full_log_ratio * response_mask).sum(dim=1) / valid_tokens_per_sample
        sample_ratio = torch.exp(masked_log_ratio)

        metrics["offpolicy/sample_importance_weight/mean"] = sample_ratio.mean().detach().item()
        metrics["offpolicy/sample_importance_weight/std"] = sample_ratio.std().detach().item()
        metrics["offpolicy/sample_importance_weight/median"] = sample_ratio.median().detach().item()
        metrics["offpolicy/sample_importance_weight/max"] = sample_ratio.max().detach().item()
        metrics["offpolicy/sample_importance_weight/min"] = sample_ratio.min().detach().item()

        # Percentile statistics for token-level
        if ratio.numel() > 0:
            metrics["offpolicy/importance_weight/percentile_05"] = torch.quantile(ratio, 0.05).detach().item()
            metrics["offpolicy/importance_weight/percentile_25"] = torch.quantile(ratio, 0.25).detach().item()
            metrics["offpolicy/importance_weight/percentile_75"] = torch.quantile(ratio, 0.75).detach().item()
            metrics["offpolicy/importance_weight/percentile_95"] = torch.quantile(ratio, 0.95).detach().item()
            metrics["offpolicy/importance_weight/percentile_99"] = torch.quantile(ratio, 0.99).detach().item()

        # Percentile statistics for sample-level
        if sample_ratio.numel() > 0:
            metrics["offpolicy/sample_importance_weight/percentile_05"] = torch.quantile(sample_ratio, 0.05).detach().item()
            metrics["offpolicy/sample_importance_weight/percentile_25"] = torch.quantile(sample_ratio, 0.25).detach().item()
            metrics["offpolicy/sample_importance_weight/percentile_75"] = torch.quantile(sample_ratio, 0.75).detach().item()
            metrics["offpolicy/sample_importance_weight/percentile_95"] = torch.quantile(sample_ratio, 0.95).detach().item()
            metrics["offpolicy/sample_importance_weight/percentile_99"] = torch.quantile(sample_ratio, 0.99).detach().item()

        # ========== Intuitive Fraction Statistics ==========
        # Token-level fractions
        metrics["offpolicy/fraction_near_one"] = ((ratio >= 0.9) & (ratio <= 1.1)).float().mean().detach().item()
        metrics["offpolicy/fraction_below_half"] = (ratio < 0.5).float().mean().detach().item()
        metrics["offpolicy/fraction_above_double"] = (ratio > 2.0).float().mean().detach().item()

        # Sample-level fractions (important for filter alignment)
        metrics["offpolicy/sample_fraction_near_one"] = ((sample_ratio >= 0.9) & (sample_ratio <= 1.1)).float().mean().detach().item()
        metrics["offpolicy/sample_fraction_below_half"] = (sample_ratio < 0.5).float().mean().detach().item()
        metrics["offpolicy/sample_fraction_above_double"] = (sample_ratio > 2.0).float().mean().detach().item()

        # Sample-level filter prediction (if ratio_clip_max is provided)
        if pg_clip is not None and hasattr(pg_clip, '__float__'):  # Check if pg_clip is actually ratio_clip_max
            # Note: pg_clip is for PPO, but we might want to track filter threshold separately
            pass

        # PPO clipping analysis with clearer naming
        if pg_clip is not None and pg_clip > 0:
            clip_low = 1 - pg_clip
            clip_high = 1 + pg_clip
            in_clip_range = (ratio >= clip_low) & (ratio <= clip_high)
            metrics["offpolicy/fraction_in_ppo_clip_range"] = in_clip_range.float().mean().detach().item()
            metrics["offpolicy/ppo_clip_threshold"] = pg_clip
            # Also keep track of how many would be clipped
            metrics["offpolicy/fraction_outside_ppo_clip_range"] = (~in_clip_range).float().mean().detach().item()

        # ========== ESS and KL Divergence ==========
        # Effective sample size (with clearer naming)
        # ESS = (sum(w))^2 / sum(w^2) where w = ratio
        ess = (ratio.sum() ** 2) / (ratio ** 2).sum()
        ess_ratio = ess / ratio.numel()  # Normalized by batch size
        metrics["offpolicy/effective_sample_size"] = ess.detach().item()
        metrics["offpolicy/effective_sample_size_ratio"] = ess_ratio.detach().item()

        # KL divergence approximation: E[log(p/q)] = E[log(ratio)]
        kl_approx = log_ratio.mean().detach().item()
        metrics["offpolicy/approx_kl_divergence"] = kl_approx

        # ========== Token Statistics ==========
        metrics["offpolicy/valid_token_count"] = valid_current.numel()
        metrics["offpolicy/total_token_count"] = response_mask.numel()
        metrics["offpolicy/token_mask_rate"] = valid_current.numel() / max(response_mask.numel(), 1)

        # ========== Store raw distribution for histogram ==========
        # Store the raw tensor for wandb histogram logging
        # This will be extracted and logged separately, not as a scalar metric
        metrics["_raw_importance_weights"] = ratio.detach().cpu()  # Token-level
        metrics["_raw_log_importance_weights"] = log_ratio.detach().cpu()  # Token-level
        metrics["_raw_sample_importance_weights"] = sample_ratio.detach().cpu()  # Sample-level

        # Add flag indicating whether log_probs were reused
        metrics["offpolicy/reused_log_probs"] = 1.0 if training_metrics is not None else 0.0

        logger.debug(
            f"Off-policy metrics computed successfully: "
            f"iw_mean={metrics['offpolicy/importance_weight/mean']:.3f}, "
            f"kl={kl_approx:.3f}, "
            f"ess_ratio={ess_ratio:.3f}, "
            f"reused_log_probs={metrics['offpolicy/reused_log_probs']}"
        )

    except Exception as e:
        logger.error(f"Failed to compute off-policy metrics: {e}")
        logger.debug(f"Error details: {str(e)}", exc_info=True)

        # Return partial metrics with error indicator
        metrics["offpolicy/error"] = 1.0
        metrics["offpolicy/error_message"] = str(e)[:100]  # Truncate error message

    return metrics


def validate_replay_batch_fields(batch: DataProto) -> Dict[str, bool]:
    """
    Validate that a replay batch contains all necessary fields for off-policy training.

    Args:
        batch: DataProto batch to validate

    Returns:
        Dictionary indicating which fields are present/valid
    """
    validation = {}

    try:
        # Check tensor batch
        if batch.batch is not None:
            validation["has_batch"] = True
            validation["has_input_ids"] = "input_ids" in batch.batch
            validation["has_response_mask"] = "response_mask" in batch.batch
            validation["has_behavior_log_probs"] = "behavior_log_probs" in batch.batch
            validation["has_old_log_probs"] = "old_log_probs" in batch.batch
            validation["has_scores"] = "scores" in batch.batch
            validation["has_penalty"] = "penalty" in batch.batch

            # Check shapes consistency
            if validation["has_input_ids"] and validation["has_response_mask"]:
                input_shape = batch.batch["input_ids"].shape
                mask_shape = batch.batch["response_mask"].shape
                validation["shape_consistent"] = (input_shape[0] == mask_shape[0])
            else:
                validation["shape_consistent"] = False
        else:
            validation["has_batch"] = False

        # Check non-tensor batch
        if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None:
            validation["has_non_tensor_batch"] = True
            validation["has_env_ids"] = "env_ids" in batch.non_tensor_batch
            validation["has_traj_id"] = "traj_id" in batch.non_tensor_batch
        else:
            validation["has_non_tensor_batch"] = False

        # Overall validity
        validation["is_valid"] = (
            validation.get("has_batch", False) and
            validation.get("has_response_mask", False) and
            (validation.get("has_behavior_log_probs", False) or validation.get("has_old_log_probs", False))
        )

    except Exception as e:
        logger.error(f"Failed to validate replay batch: {e}")
        validation["error"] = str(e)
        validation["is_valid"] = False

    return validation


def log_offpolicy_diagnostics(
    metrics: Dict[str, float],
    batch: DataProto,
    global_step: int,
    logger_func=logger.info
):
    """
    Log detailed off-policy diagnostics for debugging.

    Args:
        metrics: Off-policy metrics dictionary
        batch: The batch used for computation
        global_step: Current training step
        logger_func: Logging function to use
    """
    try:
        # Validate batch first
        validation = validate_replay_batch_fields(batch)

        # Prepare diagnostic message
        diag_parts = [
            f"[Step {global_step}] Off-Policy Diagnostics:",
            f"  Batch validation: {validation.get('is_valid', False)}",
        ]

        # Add validation details if not valid
        if not validation.get('is_valid', False):
            diag_parts.append("  Missing fields:")
            for field, present in validation.items():
                if field != "is_valid" and not present:
                    diag_parts.append(f"    - {field}")

        # Add metrics summary
        if metrics:
            diag_parts.append("  Metrics:")
            key_metrics = [
                "ratio/mean", "ratio/max", "ratio/min",
                "kl_divergence", "ess_ratio", "valid_tokens"
            ]
            for key in key_metrics:
                for full_key, value in metrics.items():
                    if key in full_key:
                        diag_parts.append(f"    {full_key}: {value:.4f}")
        else:
            diag_parts.append("  No metrics computed")

        # Log as single message
        logger_func("\n".join(diag_parts))

    except Exception as e:
        logger.error(f"Failed to log off-policy diagnostics: {e}")