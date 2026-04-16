"""
Off-policy filtering utilities for replay buffer sampling.

This module provides functions for filtering replay buffer samples based on
importance sampling ratios to ensure stable off-policy training.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from roll.distributed.scheduler.protocol import DataProto
from tensordict import TensorDict
import logging

logger = logging.getLogger(__name__)


class AdaptiveMiniBalchController:
    """
    自适应调整mini_batch_size的控制器，根据filter成功率动态调整采样批次大小。

    核心思想：
    - 成功率高 → 增大mini_batch_size → 减少采样次数，提高效率
    - 成功率低 → 保持初始mini_batch_size → 避免浪费GPU计算
    - 极限情况：成功率≥95% → 直接设为target_batch_size，一次采样成功

    使用滑动窗口平滑成功率波动：
    - 记录最近N次（window_size）的成功率
    - 根据平均成功率调整，避免单次波动导致剧烈变化

    Example:
        # 初始化控制器，初始mini_batch=32，窗口大小=5
        controller = AdaptiveMiniBalchController(initial_mini_batch_size=32, window_size=5)

        # 每次采样后更新
        for attempt in range(max_attempts):
            current_size = controller.get_current_size()  # 获取当前大小
            # ... 采样 current_size 个样本 ...
            success_rate = valid_count / current_size  # 计算成功率
            next_size = controller.update(success_rate, target_batch_size=128)  # 更新下次大小
    """

    def __init__(
        self,
        initial_mini_batch_size: int = 32,
        window_size: int = 5,
    ):
        """
        初始化自适应控制器。

        Args:
            initial_mini_batch_size: 初始mini_batch大小，也是最小值（不会低于这个值）
            window_size: 滑动窗口大小，记录最近N次成功率用于平滑
                        - window_size=1: 不平滑，完全根据最近一次
                        - window_size=5: 适中平滑（推荐）
                        - window_size=10: 强平滑，反应较慢
        """
        self.initial_size = initial_mini_batch_size  # 最小值 = 初始值
        self.current_size = initial_mini_batch_size
        self.window_size = window_size

        # 滑动窗口：存储最近N次的成功率，例如 [0.9, 0.85, 0.92, 0.88, 0.91]
        self.success_history = []

    def update(self, success_rate: float, target_batch_size: int) -> int:
        """
        根据本次成功率更新下次的mini_batch_size。

        Args:
            success_rate: 本次filter的成功率，范围[0, 1]
                         例如：采样32个，通过28个 → success_rate = 28/32 = 0.875
            target_batch_size: 目标batch大小（上限），例如128

        Returns:
            更新后的mini_batch_size，用于下次采样

        调整策略（只增不减，最小保持在initial_size）：
            - 平均成功率 ≥ 95%: 直接设为target_batch_size（一次采样）
            - 平均成功率 ≥ 90%: 翻倍增长
            - 平均成功率 ≥ 70%: 增长1.5倍
            - 平均成功率 < 70%: 回退到initial_size
        """
        # 1. 将本次成功率加入历史窗口
        self.success_history.append(success_rate)

        # 2. 保持窗口大小：如果超过window_size，删除最旧的一次
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)  # 删除第一个元素（最旧的）

        # 3. 计算滑动窗口内的平均成功率
        # 例如：history=[0.9, 0.85, 0.92] → avg=0.8567
        avg_success = sum(self.success_history) / len(self.success_history)

        # 4. 根据平均成功率调整mini_batch_size
        if avg_success >= 0.95:
            # 极高成功率：几乎所有样本都通过，直接一次采样完成
            self.current_size = target_batch_size
        elif avg_success >= 0.9:
            # 高成功率：翻倍增长，但不超过target
            self.current_size = min(self.current_size * 2, target_batch_size)
        elif avg_success >= 0.7:
            # 中等偏上成功率：适度增长1.5倍
            self.current_size = min(int(self.current_size * 1.5), target_batch_size)
        else:
            # 低成功率：回退到初始值，不再减小
            self.current_size = self.initial_size

        logger.debug(
            f"Adaptive mini_batch update: success_rate={success_rate:.2%}, "
            f"avg_success={avg_success:.2%} (window={len(self.success_history)}), "
            f"new_size={self.current_size}, target={target_batch_size}"
        )

        return self.current_size

    def get_current_size(self) -> int:
        """获取当前的mini_batch_size。"""
        return self.current_size


def filter_offpolicy_samples(
    current_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    ratio_clip_max: float = 3.0,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """
    Filter samples based on importance sampling ratio.

    Filters out samples where the importance sampling ratio (current/behavior)
    exceeds the threshold, indicating the policy has diverged too far.

    Args:
        current_log_probs: Log probs from current policy [batch_size, seq_len]
        behavior_log_probs: Log probs from behavior policy (stored in replay buffer) [batch_size, seq_len]
        response_mask: Mask for valid response tokens [batch_size, seq_len]
        ratio_clip_max: Maximum allowed importance sampling ratio
        return_stats: Whether to return filtering statistics

    Returns:
        valid_mask: Boolean mask indicating valid samples [batch_size]
        stats: Optional dictionary with filtering statistics
    """
    # Ensure all tensors are on the same device (CUDA if current_log_probs is on CUDA)
    target_device = current_log_probs.device
    if behavior_log_probs.device != target_device:
        behavior_log_probs = behavior_log_probs.to(target_device)
        logger.debug(f"Moved behavior_log_probs from {behavior_log_probs.device} to {target_device}")
    if response_mask.device != target_device:
        response_mask = response_mask.to(target_device)
        logger.debug(f"Moved response_mask from {response_mask.device} to {target_device}")

    # Handle shape mismatch: log_probs may be 1 token shorter than masks
    # This happens because log_probs are computed for next-token prediction
    if current_log_probs.shape[1] == behavior_log_probs.shape[1] - 1:
        # Pad current_log_probs with zeros for the last position
        current_log_probs = torch.nn.functional.pad(current_log_probs, (0, 1), value=0.0)
        logger.debug(f"Padded current_log_probs from {current_log_probs.shape[1]-1} to {current_log_probs.shape[1]}")

    if current_log_probs.shape[1] == response_mask.shape[1] - 1:
        # Truncate response_mask to match log_probs length
        response_mask = response_mask[:, :current_log_probs.shape[1]]
        logger.debug(f"Truncated response_mask to {response_mask.shape[1]}")

    if behavior_log_probs.shape[1] != current_log_probs.shape[1]:
        # Align behavior_log_probs with current_log_probs
        min_len = min(behavior_log_probs.shape[1], current_log_probs.shape[1])
        behavior_log_probs = behavior_log_probs[:, :min_len]
        current_log_probs = current_log_probs[:, :min_len]
        response_mask = response_mask[:, :min_len]
        logger.debug(f"Aligned all tensors to length {min_len}")

    assert current_log_probs.shape == behavior_log_probs.shape, \
        f"Shape mismatch after alignment: {current_log_probs.shape} vs {behavior_log_probs.shape}"
    assert current_log_probs.shape[1] <= response_mask.shape[1], \
        f"Shape mismatch with response_mask after alignment: {current_log_probs.shape} vs {response_mask.shape}"

    batch_size = current_log_probs.shape[0]

    # Compute importance sampling ratio using geometric mean (consistent with ROLL's seq mode)
    # Formula: ratio = exp(mean(log_ratio)) instead of mean(exp(log_ratio))
    with torch.no_grad():
        log_ratio = current_log_probs - behavior_log_probs

        # Mask out non-response tokens
        log_ratio = log_ratio * response_mask

        # Compute per-sample ratio using geometric mean (ROLL seq mode):
        # 1. First compute mean of log_ratio (in log space)
        # 2. Then exp to get the ratio
        valid_tokens = response_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        masked_log_ratio = (log_ratio * response_mask).sum(dim=1) / valid_tokens
        per_sample_ratio = torch.exp(masked_log_ratio)

        # Filter based on ratio threshold
        valid_mask = per_sample_ratio <= ratio_clip_max

    # Compute statistics if requested
    stats = None
    if return_stats:
        num_valid = valid_mask.sum().item()
        num_filtered = batch_size - num_valid

        stats = {
            "filter/total_samples": batch_size,
            "filter/valid_samples": num_valid,
            "filter/filtered_samples": num_filtered,
            "filter/filter_rate": num_filtered / batch_size if batch_size > 0 else 0.0,
            "filter/avg_ratio": per_sample_ratio.mean().item(),
            "filter/max_ratio": per_sample_ratio.max().item(),
            "filter/min_ratio": per_sample_ratio.min().item(),
            "filter/ratio_threshold": ratio_clip_max,
        }

        # Add percentile statistics
        if batch_size > 0:
            stats["filter/ratio_p50"] = per_sample_ratio.median().item()
            stats["filter/ratio_p90"] = per_sample_ratio.quantile(0.9).item()
            stats["filter/ratio_p95"] = per_sample_ratio.quantile(0.95).item()

    return valid_mask, stats


def slice_dataproto(data: DataProto, indices: torch.Tensor) -> DataProto:
    """
    Slice a DataProto batch by indices.

    Args:
        data: DataProto to slice
        indices: Indices to select [num_indices]

    Returns:
        Sliced DataProto with selected samples
    """
    if len(indices) == 0:
        # Return empty DataProto
        return DataProto(batch=TensorDict({}, batch_size=[0]), non_tensor_batch={})

    # Convert indices to list for indexing
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().tolist()

    # Slice tensor batch
    sliced_batch_dict = {}
    if data.batch is not None:
        for key, value in data.batch.items():
            if isinstance(value, torch.Tensor):
                sliced_batch_dict[key] = value[indices]
            else:
                # Handle non-tensor data (shouldn't happen in batch)
                logger.warning(f"Non-tensor data in batch['{key}'], skipping slicing")
                sliced_batch_dict[key] = value

    # Slice non-tensor batch
    sliced_non_tensor = {}
    if data.non_tensor_batch is not None:
        for key, value in data.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                # Handle numpy arrays (the standard format for non_tensor_batch in ROLL)
                sliced_values = value[indices]
                # Ensure the result is still a numpy array with dtype=object
                if not isinstance(sliced_values, np.ndarray):
                    sliced_non_tensor[key] = np.array([sliced_values], dtype=object)
                else:
                    sliced_non_tensor[key] = sliced_values
            elif isinstance(value, (list, tuple)):
                sliced_non_tensor[key] = [value[i] for i in indices]
            else:
                # Keep as is if not a sequence (scalars, etc.)
                sliced_non_tensor[key] = value

    # Create new DataProto with sliced data
    # Convert dict to TensorDict for batch
    sliced_batch = TensorDict(sliced_batch_dict, batch_size=[len(indices)]) if sliced_batch_dict else None
    result = DataProto(batch=sliced_batch, non_tensor_batch=sliced_non_tensor)

    # Copy meta_info (not sliced)
    result.meta_info = data.meta_info.copy() if data.meta_info else {}

    return result


def concatenate_dataprotos(data_list: List[DataProto]) -> DataProto:
    """
    Concatenate multiple DataProto objects along the batch dimension.

    Args:
        data_list: List of DataProto objects to concatenate

    Returns:
        Concatenated DataProto
    """
    if not data_list:
        return DataProto(batch=TensorDict({}, batch_size=[0]), non_tensor_batch={})

    if len(data_list) == 1:
        return data_list[0]

    # Concatenate tensor batches
    concat_batch_dict = {}
    all_keys = set()
    for data in data_list:
        if data.batch is not None:
            all_keys.update(data.batch.keys())

    total_batch_size = 0
    for key in all_keys:
        tensors_to_concat = []
        for data in data_list:
            if data.batch is not None and key in data.batch:
                tensors_to_concat.append(data.batch[key])

        if tensors_to_concat:
            # Ensure all tensors are on the same device
            device = tensors_to_concat[0].device
            tensors_to_concat = [t.to(device) for t in tensors_to_concat]
            concat_batch_dict[key] = torch.cat(tensors_to_concat, dim=0)
            # Track total batch size
            if total_batch_size == 0:
                total_batch_size = concat_batch_dict[key].shape[0]

    # Concatenate non-tensor batches
    concat_non_tensor = {}
    all_non_tensor_keys = set()
    for data in data_list:
        if data.non_tensor_batch is not None:
            all_non_tensor_keys.update(data.non_tensor_batch.keys())

    for key in all_non_tensor_keys:
        values_to_concat = []
        for data in data_list:
            if data.non_tensor_batch is not None and key in data.non_tensor_batch:
                value = data.non_tensor_batch[key]
                if isinstance(value, (list, tuple)):
                    # Lists/tuples: extend as individual elements
                    values_to_concat.extend(value)
                elif isinstance(value, np.ndarray):
                    # Numpy arrays: collect arrays for concatenation
                    values_to_concat.append(value)
                else:
                    # Scalars or other types: just take the first one
                    if key not in concat_non_tensor:
                        concat_non_tensor[key] = value

        # Concatenate collected values
        if values_to_concat:
            # Check if we have numpy arrays
            if isinstance(values_to_concat[0], np.ndarray):
                concat_non_tensor[key] = np.concatenate(values_to_concat, axis=0)
            else:
                # Lists/tuples already extended - convert to np.ndarray with dtype=object
                # to maintain consistency with ROLL's DataProto format
                result_array = np.empty(len(values_to_concat), dtype=object)
                result_array[:] = values_to_concat
                concat_non_tensor[key] = result_array

    # Create concatenated DataProto with TensorDict
    concat_batch = TensorDict(concat_batch_dict, batch_size=[total_batch_size]) if concat_batch_dict else None
    result = DataProto(batch=concat_batch, non_tensor_batch=concat_non_tensor)

    # Copy meta_info from first DataProto
    if data_list[0].meta_info:
        result.meta_info = data_list[0].meta_info.copy()

    return result


def filter_replay_batch_with_mini_batches(
    replay_buffer,
    actor_train,
    tokenizer,
    pipeline_config,
    target_batch_size: int = 128,
    mini_batch_size: int = 32,
    ratio_clip_max: float = 3.0,
    max_attempts: int = 20,
    global_step: int = 0,
    adaptive_mini_batch: bool = False,
) -> Tuple[Optional[DataProto], Dict[str, float]]:
    """
    Filter replay buffer samples using mini-batch forward passes to reduce memory usage.

    This function implements an optimized filtering strategy that:
    1. Samples mini-batches from replay buffer
    2. Computes current policy log_probs for each mini-batch
    3. Filters based on importance sampling ratio
    4. Stops early once enough valid samples are collected
    5. (Optional) Adaptively adjusts mini_batch_size based on filter success rate

    Args:
        replay_buffer: Replay buffer instance
        actor_train: Actor training cluster for computing log_probs
        tokenizer: Tokenizer for padding
        pipeline_config: Pipeline configuration
        target_batch_size: Number of valid samples needed
        mini_batch_size: Initial size of each mini-batch for forward pass
        ratio_clip_max: Maximum allowed importance sampling ratio
        max_attempts: Maximum number of mini-batches to try
        global_step: Current training step
        adaptive_mini_batch: Whether to enable adaptive mini_batch_size adjustment
                            If True, mini_batch_size will automatically increase based on success rate

    Returns:
        filtered_batch: DataProto with filtered samples, or None if failed
        stats: Dictionary with filtering statistics
    """
    valid_samples = []
    total_sampled = 0
    total_filtered = 0
    all_sampled_indices = []  # For PER priority update

    rb_cfg = pipeline_config.replay
    stats = {}

    # 初始化自适应控制器（如果启用）
    adaptive_controller = None
    if adaptive_mini_batch:
        # 从配置中读取窗口大小，默认5
        window_size = getattr(rb_cfg, 'filter_success_rate_window', 5)

        adaptive_controller = AdaptiveMiniBalchController(
            initial_mini_batch_size=mini_batch_size,
            window_size=window_size,
        )
        logger.info(f"Adaptive mini_batch enabled: initial={mini_batch_size}, window={window_size}")

    # 当前使用的mini_batch_size
    current_mini_batch = mini_batch_size

    logger.info(f"Starting mini-batch filtering: target={target_batch_size}, "
                f"mini_batch={current_mini_batch}, ratio_max={ratio_clip_max}, "
                f"adaptive={adaptive_mini_batch}")

    for attempt in range(max_attempts):
        # Check if we have enough valid samples (early stop)
        current_valid_count = sum(len(s.batch["input_ids"]) for s in valid_samples)
        if current_valid_count >= target_batch_size:
            logger.info(f"Early stop: collected {current_valid_count} valid samples (target: {target_batch_size})")
            break

        # 获取当前使用的mini_batch_size（如果启用自适应，则动态调整）
        if adaptive_controller is not None:
            current_mini_batch = adaptive_controller.get_current_size()

        # Sample a mini-batch from replay buffer
        sample_result = replay_buffer.sample_for_training(
            batch_size=current_mini_batch,
            device='cpu',
            tokenizer=tokenizer,
            sequence_length=pipeline_config.sequence_length,
            sampling_mode=rb_cfg.sampling_mode,
            steps_per_episode=rb_cfg.steps_per_episode,
            sample_method=getattr(rb_cfg, 'sample_method', 'uniform'),
            candidates_per_group=getattr(rb_cfg, 'candidates_per_group', 1),
            group_sampling=getattr(rb_cfg, 'group_sampling', 'uniform'),
            compute_importance_weights=getattr(rb_cfg, 'importance_sampling_correction', False),
            importance_weight_beta=getattr(rb_cfg, 'importance_beta', 0.4),
        )

        # Handle sampling result
        if sample_result is None or (isinstance(sample_result, tuple) and sample_result[0] is None):
            logger.warning(f"Replay buffer failed to sample at attempt {attempt}")
            continue

        if isinstance(sample_result, tuple):
            mb, sampled_indices = sample_result
        else:
            mb = sample_result
            sampled_indices = []

        if mb is None:
            continue

        total_sampled += current_mini_batch

        # Move to GPU and compute current policy log_probs (detached)
        mb_cuda = mb.to("cuda")
        mb_cuda.meta_info["old_prob_mode"] = "trajectory"
        mb_cuda.meta_info["is_offload_states"] = False

        try:
            # Compute log_probs using actor_train (detached, no gradients)
            with torch.no_grad():
                current_lp_refs = actor_train.compute_log_probs(mb_cuda, blocking=False)
                current_lp_data = DataProto.materialize_concat(data_refs=current_lp_refs)

            if "log_probs" not in current_lp_data.batch:
                logger.warning(f"Failed to compute log_probs at attempt {attempt}")
                del mb_cuda
                torch.cuda.empty_cache()
                continue

            current_log_probs = current_lp_data.batch["log_probs"]

            # Filter samples based on importance sampling ratio
            valid_mask, filter_stats = filter_offpolicy_samples(
                current_log_probs=current_log_probs,
                behavior_log_probs=mb_cuda.batch["old_log_probs"],
                response_mask=mb_cuda.batch["response_mask"],
                ratio_clip_max=ratio_clip_max,
                return_stats=True
            )

            # Collect valid samples
            valid_indices = torch.where(valid_mask)[0]
            num_valid = len(valid_indices)

            if num_valid > 0:
                # Slice to get valid samples and move back to CPU
                mb_valid = slice_dataproto(mb_cuda, valid_indices)
                valid_samples.append(mb_valid.to("cpu"))

                # Track indices for PER update if applicable
                if sampled_indices:
                    valid_sampled_indices = [sampled_indices[i.item()] for i in valid_indices]
                    all_sampled_indices.extend(valid_sampled_indices)

            total_filtered += (current_mini_batch - num_valid)

            # 更新自适应控制器（如果启用）
            if adaptive_controller is not None:
                success_rate = num_valid / current_mini_batch if current_mini_batch > 0 else 0.0
                next_size = adaptive_controller.update(success_rate, target_batch_size)
                logger.info(f"Mini-batch {attempt}: {num_valid}/{current_mini_batch} valid "
                           f"(success={success_rate:.1%}, avg_ratio={filter_stats['filter/avg_ratio']:.2f}), "
                           f"next_size={next_size}")
            else:
                logger.debug(f"Mini-batch {attempt}: {num_valid}/{current_mini_batch} valid "
                            f"(avg_ratio={filter_stats['filter/avg_ratio']:.2f}, "
                            f"max_ratio={filter_stats['filter/max_ratio']:.2f})")

        finally:
            # Clean up GPU memory
            del mb_cuda
            if 'current_log_probs' in locals():
                del current_log_probs
            if 'current_lp_data' in locals():
                del current_lp_data
            torch.cuda.empty_cache()

    # Check if we collected any valid samples
    if not valid_samples:
        logger.warning(f"Failed to collect any valid samples after {max_attempts} attempts. "
                      f"Falling back to unfiltered sampling.")

        # Fallback: 直接采样目标数量，不进行过滤
        fallback_result = replay_buffer.sample_for_training(
            batch_size=target_batch_size,
            device='cpu',
            tokenizer=tokenizer,
            sequence_length=pipeline_config.sequence_length,
            sampling_mode=rb_cfg.sampling_mode,
            steps_per_episode=rb_cfg.steps_per_episode,
            sample_method=getattr(rb_cfg, 'sample_method', 'uniform'),
            candidates_per_group=getattr(rb_cfg, 'candidates_per_group', 1),
            group_sampling=getattr(rb_cfg, 'group_sampling', 'uniform'),
            compute_importance_weights=getattr(rb_cfg, 'importance_sampling_correction', False),
            importance_weight_beta=getattr(rb_cfg, 'importance_beta', 0.4),
        )

        stats = {
            "filter/total_sampled": total_sampled,
            "filter/total_valid": 0,
            "filter/total_filtered": total_sampled,
            "filter/filter_rate": 1.0,
            "filter/attempts": max_attempts,
            "filter/success": False,
            "filter/fallback": True,  # 标记使用了fallback
        }

        if fallback_result is None:
            logger.error("Even fallback sampling failed")
            return None, stats

        # Handle return format
        if isinstance(fallback_result, tuple):
            fallback_batch, fallback_indices = fallback_result
            return (fallback_batch, fallback_indices), stats
        else:
            return fallback_result, stats

    # Concatenate all valid samples
    final_batch = concatenate_dataprotos(valid_samples)
    final_size = len(final_batch.batch["input_ids"])

    # Handle insufficient samples
    if final_size < target_batch_size:
        min_acceptable = getattr(rb_cfg, 'filter_min_acceptable_batch', target_batch_size // 2)

        if final_size < min_acceptable:
            # Too few samples, need to supplement or fallback
            logger.warning(f"Only collected {final_size} valid samples (min: {min_acceptable}). "
                          f"Supplementing with unfiltered samples.")

            # 补充采样：采样额外的不过滤样本来补齐
            supplement_size = target_batch_size - final_size
            supplement_result = replay_buffer.sample_for_training(
                batch_size=supplement_size,
                device='cpu',
                tokenizer=tokenizer,
                sequence_length=pipeline_config.sequence_length,
                sampling_mode=rb_cfg.sampling_mode,
                steps_per_episode=rb_cfg.steps_per_episode,
                sample_method=getattr(rb_cfg, 'sample_method', 'uniform'),
                candidates_per_group=getattr(rb_cfg, 'candidates_per_group', 1),
                group_sampling=getattr(rb_cfg, 'group_sampling', 'uniform'),
            )

            if supplement_result is not None:
                if isinstance(supplement_result, tuple):
                    supplement_batch, supplement_indices = supplement_result
                    if supplement_indices and all_sampled_indices:
                        all_sampled_indices.extend(supplement_indices)
                else:
                    supplement_batch = supplement_result

                # 合并过滤后的样本和补充样本
                valid_samples.append(supplement_batch)
                final_batch = concatenate_dataprotos(valid_samples)
                final_size = len(final_batch.batch["input_ids"])

                logger.info(f"Supplemented to {final_size} samples (filtered: {final_size - supplement_size}, "
                           f"unfiltered: {supplement_size})")
        else:
            # 样本数量在可接受范围内，直接使用
            logger.info(f"Using {final_size} valid samples (target was {target_batch_size})")

    # If we have more than needed, randomly select target_batch_size samples
    if final_size > target_batch_size:
        indices = torch.randperm(final_size)[:target_batch_size]
        final_batch = slice_dataproto(final_batch, indices)

        # Update sampled indices for PER
        if all_sampled_indices:
            all_sampled_indices = [all_sampled_indices[i.item()] for i in indices]

    # Compute final statistics
    actual_size = len(final_batch.batch["input_ids"])
    stats = {
        "filter/total_sampled": total_sampled,
        "filter/total_valid": actual_size,
        "filter/total_filtered": total_filtered,
        "filter/filter_rate": total_filtered / total_sampled if total_sampled > 0 else 0.0,
        "filter/attempts": attempt + 1,
        "filter/mini_batches": attempt + 1,
        "filter/early_stop": current_valid_count >= target_batch_size,
        "filter/success": True,
        "filter/supplemented": final_size < target_batch_size,  # 标记是否补充了样本
        "filter/adaptive_enabled": adaptive_mini_batch,
    }

    # 添加自适应控制器的统计信息
    if adaptive_controller is not None:
        stats["filter/final_mini_batch_size"] = adaptive_controller.get_current_size()
        if adaptive_controller.success_history:
            stats["filter/avg_success_rate"] = sum(adaptive_controller.success_history) / len(adaptive_controller.success_history)

    logger.info(f"Filtering complete: sampled={total_sampled}, valid={actual_size}, "
               f"filtered={total_filtered} ({stats['filter/filter_rate']:.1%}), "
               f"attempts={attempt + 1}")

    # Return with sampled indices for PER if applicable
    if all_sampled_indices:
        return (final_batch, all_sampled_indices), stats
    else:
        return final_batch, stats