"""
Trajectory-Level Replay Buffer for ROLL Framework

Specifically designed to work with TrajEnvManager data format.
Stores complete episodes as single units with episode-level rewards and penalties.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import random
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from .base_buffer import BaseReplayBuffer
from .segment_tree import SumSegmentTree, MinSegmentTree, next_power_of_2

logger = get_logger()


@dataclass
class TrajectoryEntry:
    """
    Single trajectory entry for trajectory-level replay buffer.
    Matches the output format of TrajEnvManager.
    """
    # Core data from DataProto
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    response_mask: np.ndarray
    prompt_mask: np.ndarray
    scores: np.ndarray  # Token-level scores with episode reward on last response token
    penalty: float      # Episode-level penalty scalar
    behavior_log_probs: np.ndarray  # Log probabilities from behavior policy for off-policy analysis
    
    # Metadata from non_tensor_batch
    env_id: str
    group_id: str
    messages_list: List[Dict]
    tag: str
    frames: List
    step_scores: List
    episode_scores: List
    traj_group_id: str
    traj_id: str
    
    # Storage metadata
    stored_at_step: int
    episode_length: int

    # Priority-related metadata
    priority: float = 1.0       # Current priority value (intrinsic value)
    sample_count: int = 0       # Number of times sampled (for statistics)
    global_step: int = 0        # Global training step when stored (for age calculation)


class TrajectoryReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer specialized for trajectory-level data from TrajEnvManager.
    
    Key Design Principles:
    1. Episode-level storage: complete trajectories as atomic units
    2. TrajEnvManager compatibility: matches exact data format
    3. Episode-level penalty handling: penalties are episode totals, not step-wise
    4. Efficient trajectory sampling: maintains episode boundaries
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 128,
        seed: int = 42,
        priority_fn: callable = None,
        priority_exponent: float = 0.6,
        age_decay: float = 1000.0,
        enable_age_decay: bool = False,
        eviction_strategy: str = "fifo"
    ):
        super().__init__(capacity, batch_size, seed)
        self.trajectories = [None] * capacity
        self.valid_mask = [False] * capacity
        self.num_valid = 0
        self.rng = random.Random(seed)
        self.eviction_strategy = eviction_strategy.lower()
        self.enable_smart_eviction = (self.eviction_strategy == "smart")

        # Priority configuration
        from .priority_functions import uniform_priority
        self.priority_fn = priority_fn or uniform_priority
        self.priority_exponent = priority_exponent

        # Segment Tree for O(log n) prioritized sampling
        self._tree_capacity = next_power_of_2(capacity)
        self._it_sum = SumSegmentTree(self._tree_capacity)
        self._it_min = MinSegmentTree(self._tree_capacity)
        self._max_priority = 1.0

        # Age-based freshness weighting (optional, default off for standard PER behavior)
        self.enable_age_decay = enable_age_decay
        self.age_decay = age_decay
        self.current_global_step = 0

        logger.info(
            f"TrajectoryReplayBuffer: capacity={capacity}, priority_fn={self.priority_fn.__name__}, "
            f"priority_exponent={priority_exponent}, enable_age_decay={enable_age_decay}, "
            f"age_decay={age_decay}, eviction={eviction_strategy}"
        )
    
    @property
    def buffer_type(self) -> str:
        return "trajectory"
    
    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store trajectory data from TrajEnvManager.

        Args:
            batch: DataProto from TrajEnvManager containing complete episodes
            global_step: Current training step
        """
        # Update current global step for age calculation
        self.current_global_step = global_step

        batch_size = batch.batch["input_ids"].shape[0]
        
        for i in range(batch_size):
            # Extract tensor data
            input_ids = batch.batch["input_ids"][i].cpu().numpy()
            attention_mask = batch.batch["attention_mask"][i].cpu().numpy()
            position_ids = batch.batch["position_ids"][i].cpu().numpy()
            response_mask = batch.batch["response_mask"][i].cpu().numpy()
            prompt_mask = batch.batch["prompt_mask"][i].cpu().numpy()
            scores = batch.batch["scores"][i].cpu().numpy()
            penalty = float(batch.batch["penalty"][i].cpu().item())
            
            # Extract behavior policy log_probs if available
            behavior_log_probs = None
            if "behavior_log_probs" in batch.batch:
                behavior_log_probs = batch.batch["behavior_log_probs"][i].cpu().numpy()
            else:
                # If no behavior_log_probs, create zeros with next-token length (len(input_ids)-1)
                target_len = max(int(input_ids.shape[0]) - 1, 0)
                behavior_log_probs = np.zeros((target_len,), dtype=np.float32)
            
            # Extract metadata
            env_id = batch.non_tensor_batch["env_ids"][i]
            group_id = batch.non_tensor_batch["group_ids"][i]
            messages_list = batch.non_tensor_batch["messages_list"][i]
            tag = batch.non_tensor_batch["tags"][i]
            frames = batch.non_tensor_batch["frames"][i]
            step_scores = batch.non_tensor_batch["step_scores"][i]
            episode_scores = batch.non_tensor_batch["episode_scores"][i]
            traj_group_id = batch.non_tensor_batch["traj_group_id"][i]
            traj_id = batch.non_tensor_batch["traj_id"][i]
            
            # Calculate episode length from attention mask
            episode_length = int(attention_mask.sum())
            
            trajectory = TrajectoryEntry(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                response_mask=response_mask,
                prompt_mask=prompt_mask,
                scores=scores,
                penalty=penalty,
                behavior_log_probs=behavior_log_probs,
                env_id=env_id,
                group_id=group_id,
                messages_list=messages_list,
                tag=tag,
                frames=frames,
                step_scores=step_scores,
                episode_scores=episode_scores,
                traj_group_id=traj_group_id,
                traj_id=traj_id,
                stored_at_step=global_step,
                episode_length=episode_length,
                global_step=global_step  # Store global step for age calculation
            )

            # Calculate priority for this trajectory
            # Pass age_decay as kwarg for reward_fresh and other age-aware priority functions
            try:
                priority = self.priority_fn(trajectory, global_step, age_decay=self.age_decay)
                trajectory.priority = float(priority)
            except Exception as e:
                logger.warning(f"Failed to calculate priority, using default 1.0: {e}")
                trajectory.priority = 1.0

            # Find slot for new trajectory
            if self.num_valid >= self.capacity:
                # Buffer full: smart eviction to find slot
                slot_idx = self._smart_evict_and_get_slot()
            else:
                # Buffer not full: find empty slot
                slot_idx = self._find_empty_slot()
                self.num_valid += 1

            # Store trajectory in the slot
            self.trajectories[slot_idx] = trajectory
            self.valid_mask[slot_idx] = True

            # Update segment trees with max_priority^alpha (standard PER convention)
            # New samples get max_priority to ensure they're sampled at least once
            # The actual priority will be updated after training via update_priorities()
            priority_alpha = self._max_priority ** self.priority_exponent
            self._it_sum[slot_idx] = priority_alpha
            self._it_min[slot_idx] = priority_alpha

            self.total_stored += 1

        # Periodic garbage collection to prevent memory leaks
        if self.total_stored % 100000 == 0 and self.total_stored > 0:
            import gc
            gc.collect()
            logger.info(f"Replay buffer GC triggered at {self.total_stored} trajectories stored")

        logger.debug(f"Stored {batch_size} trajectories. Total valid: {self.num_valid}, Total stored: {self.total_stored}")

    def _find_empty_slot(self) -> int:
        """Find the first empty slot in the buffer."""
        for i in range(self.capacity):
            if not self.valid_mask[i]:
                return i
        # Should not reach here if called correctly
        raise RuntimeError("No empty slot found but buffer reports not full")

    def _smart_evict_and_get_slot(self) -> int:
        """
        Intelligently evict a trajectory based on age and priority.

        The eviction score combines:
        - Age: Older trajectories are more likely to be evicted
        - Priority: Lower priority trajectories are more likely to be evicted
        - Sampling frequency: Over-sampled trajectories may be evicted

        Formula: eviction_score = (age / age_decay) / (priority + epsilon)
        Higher score = more likely to be evicted

        Returns:
            The slot index that was evicted and can be reused
        """
        if not self.enable_smart_eviction:
            # Fallback to FIFO: find oldest slot
            oldest_idx = 0
            oldest_step = float('inf')
            for i in range(self.capacity):
                if self.valid_mask[i] and self.trajectories[i].global_step < oldest_step:
                    oldest_step = self.trajectories[i].global_step
                    oldest_idx = i
            return oldest_idx

        # Calculate eviction scores for all valid trajectories
        eviction_scores = []
        epsilon = 1e-6  # Prevent division by zero

        for i in range(self.capacity):
            if not self.valid_mask[i]:
                continue  # Skip empty slots

            traj = self.trajectories[i]

            # Age factor (higher age = higher score = more likely to evict)
            age = self.current_global_step - traj.global_step
            age_factor = age / self.age_decay if self.age_decay > 0 else age

            # Priority factor from segment tree (lower priority = higher score)
            priority = self._it_sum[i] if i < self._tree_capacity else epsilon
            priority_factor = 1.0 / (priority + epsilon)

            # Sample count factor (over-sampled = higher score)
            sample_factor = np.log1p(traj.sample_count) / 10.0  # Logarithmic scaling

            # Combined eviction score
            eviction_score = age_factor * priority_factor * (1.0 + sample_factor)
            eviction_scores.append((i, eviction_score, age, priority))

        # Select trajectory with highest eviction score
        idx_to_evict, score, age, priority = max(eviction_scores, key=lambda x: x[1])

        # Log eviction decision (only occasionally to avoid spam)
        if self.total_stored % 100 == 0:
            logger.debug(
                f"Smart eviction: slot={idx_to_evict}, score={score:.4f}, "
                f"age={age}, priority={priority:.4f}, sample_count={self.trajectories[idx_to_evict].sample_count}"
            )

        # Clear segment tree entries for this slot (will be overwritten)
        self._it_sum[idx_to_evict] = 0.0
        self._it_min[idx_to_evict] = float('inf')

        return idx_to_evict

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough trajectories for sampling."""
        required_size = batch_size or self.batch_size
        return self.num_valid >= required_size
    
    def sample_for_training(
        self,
        batch_size: Optional[int] = None,
        device: str = 'cpu',
        tokenizer: Optional[PreTrainedTokenizer] = None,
        sequence_length: int = 4096,
        sampling_mode: str = "trajectory",
        steps_per_episode: int = 1,
        sample_method: str = "uniform",
        candidates_per_group: int = 1,
        group_sampling: str = "uniform",
        compute_importance_weights: bool = False,
        importance_weight_beta: float = 0.4
    ) -> Optional[Tuple[DataProto, List[int]]]:
        """
        Sample trajectories and reconstruct DataProto format.

        Args:
            batch_size: Number of trajectories to sample
            device: Target device for tensors ('cpu' or 'cuda')
            tokenizer: Tokenizer for text processing
            sequence_length: Maximum sequence length for padding/truncation
            sampling_mode: "trajectory" or "step" sampling mode (ignored, always trajectory)
            steps_per_episode: Number of steps per episode (ignored for trajectory mode)
            sample_method: Sampling method ("uniform", "weighted", etc.)
            candidates_per_group: Number of candidates per group
            group_sampling: Group sampling strategy ("uniform", etc.)
            compute_importance_weights: Whether to compute importance weights for PER
            importance_weight_beta: Beta parameter for importance weight (0.4 -> 1.0 annealing)

        Returns:
            Tuple of (DataProto batch, sampled_indices)
            - DataProto contains training batch with optional importance_weights
            - sampled_indices: list of buffer indices for priority update after training
        """
        sample_size = batch_size or self.batch_size

        if not self.can_sample(sample_size):
            logger.debug(f"Insufficient trajectories for sampling: {self.num_valid} < {sample_size}")
            return None, []

        # Sample trajectories based on priority function
        # Deterministic strategies: uniform, lifo, fifo
        # Weighted strategies: reward, td_error, recency, combined, etc.
        # Build list of valid trajectories
        buffer_list = []
        valid_indices = []  # Map from buffer_list index to slot index
        for i in range(self.capacity):
            if self.valid_mask[i]:
                buffer_list.append(self.trajectories[i])
                valid_indices.append(i)
        buffer_size = len(buffer_list)
        priority_fn_name = self.priority_fn.__name__

        # Track sampled indices for priority updates and importance weights
        sampled_indices = []

        if priority_fn_name == "lifo_priority":
            # LIFO (Last-In-First-Out): Deterministic sampling of newest N trajectories
            # Recommended for Echo mode (train_steps_per_env_step=1) for near-on-policy training
            start_idx = max(0, buffer_size - sample_size)
            buffer_indices = list(range(start_idx, buffer_size))
            sampled_trajectories = buffer_list[start_idx:]
            sampled_indices = [valid_indices[i] for i in buffer_indices]  # Convert to slot indices
            logger.debug(f"LIFO sampling: selected last {len(sampled_trajectories)} trajectories")

        elif priority_fn_name == "fifo_priority":
            # FIFO (First-In-First-Out): Deterministic sampling of oldest N trajectories
            # Ensures all data is used before eviction
            buffer_indices = list(range(sample_size))
            sampled_trajectories = buffer_list[:sample_size]
            sampled_indices = [valid_indices[i] for i in buffer_indices]  # Convert to slot indices
            logger.debug(f"FIFO sampling: selected first {len(sampled_trajectories)} trajectories")

        elif priority_fn_name == "uniform_priority":
            # Uniform random sampling: Standard replay buffer behavior
            buffer_indices = self.rng.sample(range(buffer_size), sample_size)
            sampled_trajectories = [buffer_list[i] for i in buffer_indices]
            sampled_indices = [valid_indices[i] for i in buffer_indices]  # Convert to slot indices
            logger.debug(f"Uniform sampling: randomly selected {len(sampled_trajectories)} trajectories")

        else:
            # Weighted priority-based sampling using Segment Tree (O(log n) per sample)
            # This is the core of Prioritized Experience Replay (PER)
            # Note: _sample_proportional now needs to work with valid slots only
            slot_indices = self._sample_proportional_slots(sample_size)
            sampled_trajectories = [self.trajectories[i] for i in slot_indices]
            sampled_indices = slot_indices

            # Update sample counts for statistics
            for idx in sampled_indices:
                self.trajectories[idx].sample_count += 1

            logger.debug(f"PER sampling ({priority_fn_name}): sampled {len(sampled_trajectories)} trajectories with priority_alpha={self.priority_exponent}")
        
        # Use pipeline's sequence_length for consistent padding (like RolloutScheduler)
        # This ensures compatibility with original ROLL behavior and prevents length mismatch
        max_seq_len = sequence_length
        
        # Prepare tensors
        target_device = torch.device(device if device is not None else 'cpu')
        
        batch_input_ids = torch.zeros((sample_size, max_seq_len), dtype=torch.long, device=target_device)
        batch_attention_mask = torch.zeros((sample_size, max_seq_len), dtype=torch.bool, device=target_device)
        batch_position_ids = torch.zeros((sample_size, max_seq_len), dtype=torch.long, device=target_device)
        batch_response_mask = torch.zeros((sample_size, max_seq_len), dtype=torch.bool, device=target_device)
        batch_prompt_mask = torch.zeros((sample_size, max_seq_len), dtype=torch.bool, device=target_device)
        batch_scores = torch.zeros((sample_size, max_seq_len), dtype=torch.float, device=target_device)
        batch_penalties = torch.zeros(sample_size, dtype=torch.float, device=target_device)
        # old_log_probs follow next-token semantics: length is (sequence_length - 1)
        batch_old_log_probs = torch.zeros((sample_size, max_seq_len - 1), dtype=torch.float, device=target_device)
        
        # Padding values (consistent with RolloutScheduler._apply_pipeline_padding)
        pad_token_id = tokenizer.pad_token_id if tokenizer else 0
        
        # Prepare non-tensor batch
        env_ids = []
        group_ids = []
        messages_lists = []
        tags = []
        frames_lists = []
        step_scores_lists = []
        episode_scores_lists = []
        traj_group_ids = []
        traj_ids = []
        
        for i, traj in enumerate(sampled_trajectories):
            original_seq_len = len(traj.input_ids)
            
            # Apply consistent truncation/padding like RolloutScheduler
            effective_seq_len = min(original_seq_len, max_seq_len)
            
            # Use pad_to_length for consistent behavior with original ROLL env_manager
            traj_input_ids = torch.from_numpy(traj.input_ids)
            traj_attention_mask = torch.from_numpy(traj.attention_mask.astype(bool))
            traj_position_ids = torch.from_numpy(traj.position_ids)
            traj_response_mask = torch.from_numpy(traj.response_mask.astype(bool))
            traj_prompt_mask = torch.from_numpy(traj.prompt_mask.astype(bool))
            traj_scores = torch.from_numpy(traj.scores)
            
            batch_input_ids[i] = pad_to_length(traj_input_ids, max_seq_len, pad_token_id)
            # Keep attention_mask padded with 0 to match RolloutScheduler
            batch_attention_mask[i] = pad_to_length(traj_attention_mask, max_seq_len, 0)
            batch_position_ids[i] = pad_to_length(traj_position_ids, max_seq_len, 0)
            batch_response_mask[i] = pad_to_length(traj_response_mask, max_seq_len, 0)
            batch_prompt_mask[i] = pad_to_length(traj_prompt_mask, max_seq_len, 0)
            batch_scores[i] = pad_to_length(traj_scores, max_seq_len, 0.0)
            
            batch_penalties[i] = traj.penalty
            
            # Handle behavior_log_probs with pad_to_length to next-token length (sequence_length-1)
            traj_behavior_log_probs = torch.from_numpy(traj.behavior_log_probs)
            batch_old_log_probs[i] = pad_to_length(traj_behavior_log_probs, max_seq_len - 1, 0.0)
            
            # Collect non-tensor data
            env_ids.append(traj.env_id)
            group_ids.append(traj.group_id)
            messages_lists.append(traj.messages_list)
            tags.append(traj.tag)
            frames_lists.append(traj.frames)
            step_scores_lists.append(traj.step_scores)
            episode_scores_lists.append(traj.episode_scores)
            traj_group_ids.append(traj.traj_group_id)
            traj_ids.append(traj.traj_id)
        
        # Create DataProto in the exact format TrajEnvManager produces
        dataproto = DataProto()
        dataproto.batch = TensorDict({
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "position_ids": batch_position_ids,
            "response_mask": batch_response_mask,
            "prompt_mask": batch_prompt_mask,
            "scores": batch_scores,
            "penalty": batch_penalties,
            "old_log_probs": batch_old_log_probs,
        }, batch_size=[sample_size])
        
        dataproto.non_tensor_batch = {
            "env_ids": np.array(env_ids, dtype=object),
            "group_ids": np.array(group_ids, dtype=object),
            "messages_list": np.array(messages_lists, dtype=object),
            "tags": np.array(tags, dtype=object),
            "frames": np.array(frames_lists, dtype=object),
            "step_scores": np.array(step_scores_lists, dtype=object),
            "episode_scores": np.array(episode_scores_lists, dtype=object),
            "traj_group_id": np.array(traj_group_ids, dtype=object),
            "traj_id": np.array(traj_ids, dtype=object),
        }
        
        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "trajectory",
            "sample_size": sample_size,
            "buffer_utilization": self.num_valid / self.capacity,
            "sampled_indices": sampled_indices  # For priority update after training
        }

        # Compute importance weights for PER (off-policy correction)
        if compute_importance_weights and priority_fn_name not in ["lifo_priority", "fifo_priority", "uniform_priority"]:
            importance_weights = self.compute_importance_weights(sampled_indices, beta=importance_weight_beta)
            dataproto.batch["importance_weights"] = torch.from_numpy(importance_weights).to(target_device)
            logger.debug(f"Computed importance weights with beta={importance_weight_beta:.2f}, mean={importance_weights.mean():.4f}")

        logger.debug(f"Sampled {sample_size} trajectories for training (indices: {len(sampled_indices)})")
        return dataproto, sampled_indices

    def _sample_proportional_slots(self, batch_size: int) -> List[int]:
        """
        Sample slot indices based on priorities using Segment Tree.
        Works directly with fixed slots and valid_mask.

        Time complexity: O(batch_size * log n)

        Args:
            batch_size: Number of samples to draw

        Returns:
            List of sampled slot indices (not buffer_list indices)
        """
        # Calculate total priority from valid slots only
        p_total = 0.0
        for i in range(self.capacity):
            if self.valid_mask[i]:
                p_total += self._it_sum[i]

        if p_total <= 0:
            # Fallback to uniform if no valid priorities
            logger.warning("Total priority is 0, falling back to uniform sampling")
            valid_slots = [i for i in range(self.capacity) if self.valid_mask[i]]
            return self.rng.sample(valid_slots, batch_size)

        indices = []
        # Stratified sampling: divide into batch_size segments
        every_range_len = p_total / batch_size

        for i in range(batch_size):
            # Sample uniformly within this priority range
            mass = self.rng.random() * every_range_len + i * every_range_len

            # Find slot index with cumulative sum search
            idx = self._find_slot_by_cumsum(mass)

            # Ensure we have a valid slot
            if idx is not None and self.valid_mask[idx]:
                indices.append(idx)
            else:
                # Fallback to random valid slot if something goes wrong
                valid_slots = [j for j in range(self.capacity) if self.valid_mask[j]]
                if valid_slots:
                    indices.append(self.rng.choice(valid_slots))

        return indices

    def _find_slot_by_cumsum(self, target_sum: float) -> Optional[int]:
        """Find slot index where cumulative sum exceeds target."""
        cumsum = 0.0
        for i in range(self.capacity):
            if self.valid_mask[i]:
                cumsum += self._it_sum[i]
                if cumsum >= target_sum:
                    return i
        return None

    def _sample_proportional(self, batch_size: int, buffer_size: int) -> List[int]:
        """
        Sample indices based on priorities using Segment Tree.

        This implements stratified sampling from Prioritized Experience Replay:
        - Divide total priority into batch_size equal ranges
        - Sample uniformly within each range
        - Use SumSegmentTree.find_prefixsum_idx for O(log n) lookup

        Time complexity: O(batch_size * log n)

        Args:
            batch_size: Number of samples to draw
            buffer_size: Current buffer size

        Returns:
            List of sampled indices
        """
        indices = []
        p_total = self._it_sum.sum(0, buffer_size)

        if p_total <= 0:
            # Fallback to uniform if no valid priorities
            logger.warning("Total priority is 0, falling back to uniform sampling")
            return self.rng.sample(range(buffer_size), batch_size)

        # Stratified sampling: divide into batch_size segments
        every_range_len = p_total / batch_size

        for i in range(batch_size):
            # Sample uniformly within this segment
            mass = self.rng.random() * every_range_len + i * every_range_len
            # Find the index corresponding to this priority mass
            idx = self._it_sum.find_prefixsum_idx(mass)
            # Ensure index is within buffer bounds
            idx = min(idx, buffer_size - 1)
            indices.append(idx)

        return indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray, current_global_step: Optional[int] = None) -> None:
        """
        Update priorities for sampled trajectories after training (standard PER).

        Standard PER formula:
            tree[idx] = priority^α
            max_priority = max(max_priority, priority)

        Optional age decay (when enable_age_decay=True):
            effective_priority = priority × exp(-age / age_decay)
            tree[idx] = effective_priority^α

        Args:
            indices: Buffer indices of sampled trajectories
            priorities: New priority values (e.g., |advantage|, |TD-error|)
            current_global_step: Current training step for age calculation (only used if enable_age_decay=True)
        """
        assert len(indices) == len(priorities), \
            f"Indices and priorities length mismatch: {len(indices)} vs {len(priorities)}"

        global_step = current_global_step if current_global_step is not None else self.current_global_step

        for slot_idx, priority in zip(indices, priorities):
            if not (0 <= slot_idx < self.capacity) or not self.valid_mask[slot_idx]:
                logger.warning(f"Invalid slot {slot_idx}, skipping priority update")
                continue

            # Ensure priority is positive
            priority = max(float(priority), 1e-6)

            # Update stored priority in trajectory
            self.trajectories[slot_idx].priority = priority

            # Compute effective priority (with optional age decay)
            if self.enable_age_decay:
                sample_age = global_step - self.trajectories[slot_idx].global_step
                freshness_weight = np.exp(-sample_age / self.age_decay)
                effective_priority = priority * freshness_weight
            else:
                effective_priority = priority

            # Update segment trees with priority^alpha (standard PER)
            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[slot_idx] = priority_alpha
            self._it_min[slot_idx] = priority_alpha

            # Track maximum priority
            self._max_priority = max(self._max_priority, priority)

        logger.debug(f"Updated priorities for {len(indices)} trajectories. "
                    f"Max priority: {self._max_priority:.4f}, age_decay={self.enable_age_decay}")

    def get_effective_priority(self, idx: int, current_global_step: Optional[int] = None) -> float:
        """
        Compute effective priority for a given buffer index.

        If enable_age_decay=True:
            effective_priority = priority × exp(-age / age_decay)
        Otherwise:
            effective_priority = priority

        Args:
            idx: Buffer index
            current_global_step: Current training step (only used if enable_age_decay=True)

        Returns:
            Effective priority value
        """
        if not (0 <= idx < self.capacity) or not self.valid_mask[idx]:
            return 0.0

        priority = self.trajectories[idx].priority

        if self.enable_age_decay:
            global_step = current_global_step if current_global_step is not None else self.current_global_step
            sample_age = global_step - self.trajectories[idx].global_step
            freshness_weight = np.exp(-sample_age / self.age_decay)
            return priority * freshness_weight

        return priority

    def refresh_all_age_decay(self, current_global_step: int) -> int:
        """
        Refresh age decay for ALL trajectories in the buffer, updating segment tree.

        This method should be called periodically (e.g., every training step) to ensure
        that old trajectories have their priorities properly decayed based on their age.

        Without this refresh, only sampled entries get their age decay updated via
        update_priorities(), leaving unsampled old entries with stale (too high) priorities.

        Time complexity: O(capacity)
        Typical runtime: ~50-100ms for capacity=100,000

        This method is designed to be called asynchronously during GPU training,
        so the CPU overhead is hidden by GPU computation time.

        Args:
            current_global_step: Current training step for age calculation

        Returns:
            Number of trajectories refreshed

        Example:
            >>> # Call during GPU training (async)
            >>> from concurrent.futures import ThreadPoolExecutor
            >>> executor = ThreadPoolExecutor(max_workers=1)
            >>> future = executor.submit(buffer.refresh_all_age_decay, global_step)
            >>> # ... GPU training happens here ...
            >>> refreshed_count = future.result()  # Wait before sampling
        """
        if not self.enable_age_decay:
            logger.debug("Age decay not enabled, skipping refresh")
            return 0

        # Update current global step
        self.current_global_step = current_global_step

        refreshed_count = 0
        max_age = 0
        total_age = 0

        for idx in range(self.capacity):
            if not self.valid_mask[idx]:
                continue

            trajectory = self.trajectories[idx]
            if trajectory is None:
                continue

            # Calculate current age and freshness
            age = max(0, current_global_step - trajectory.global_step)
            freshness_weight = np.exp(-age / self.age_decay)

            # Calculate effective priority with age decay
            effective_priority = max(trajectory.priority * freshness_weight, 1e-8)

            # Update segment tree
            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[idx] = priority_alpha
            self._it_min[idx] = priority_alpha

            refreshed_count += 1
            max_age = max(max_age, age)
            total_age += age

        avg_age = total_age / refreshed_count if refreshed_count > 0 else 0

        logger.debug(
            f"[AGE_DECAY] Refreshed {refreshed_count} trajectories at step {current_global_step}. "
            f"avg_age={avg_age:.1f}, max_age={max_age}, age_decay={self.age_decay}"
        )

        return refreshed_count

    @staticmethod
    def compute_advantage_priorities(batch: DataProto) -> np.ndarray:
        """
        Compute advantage-based priorities from a DataProto batch.

        This extracts advantages from the batch and computes mean absolute advantage
        per trajectory, masked by response tokens.

        Args:
            batch: DataProto containing "advantages" and "response_mask"

        Returns:
            priorities: [batch_size] array of advantage-based priorities

        Example:
            >>> # After compute_advantage() in pipeline
            >>> priorities = TrajectoryReplayBuffer.compute_advantage_priorities(batch)
            >>> replay_buffer.update_priorities(sampled_indices, priorities, global_step)
        """
        if "advantages" not in batch.batch or "response_mask" not in batch.batch:
            raise ValueError("Batch must contain 'advantages' and 'response_mask' for advantage-based priorities")

        advantages = batch.batch["advantages"]  # [batch_size, seq_len]
        response_mask = batch.batch["response_mask"]  # [batch_size, seq_len]

        # Compute masked mean absolute advantage per sample
        abs_advantages = torch.abs(advantages)
        masked_advantages = abs_advantages * response_mask.float()

        # Sum over tokens and divide by number of response tokens
        sum_advantages = masked_advantages.sum(dim=1)  # [batch_size]
        num_tokens = response_mask.sum(dim=1).float().clamp(min=1.0)  # [batch_size], avoid division by zero

        priorities = (sum_advantages / num_tokens).cpu().numpy()

        return priorities

    def compute_importance_weights(
        self,
        indices: List[int],
        beta: float = 0.4
    ) -> np.ndarray:
        """
        Compute importance sampling weights for off-policy correction.

        Importance weights correct for the bias introduced by prioritized sampling.
        Formula: w_i = (N * P(i))^(-beta) / max_w

        Beta annealing schedule (common in PER):
        - Start: beta = 0.4 (partial correction)
        - End: beta = 1.0 (full correction)
        - Anneal linearly over training

        Args:
            indices: Buffer indices of sampled trajectories
            beta: Importance weight exponent (0 = no correction, 1 = full correction)

        Returns:
            Importance weights normalized by max weight, shape [batch_size]

        Example:
            >>> indices, weights = buffer.sample_with_weights(batch_size=128, beta=0.6)
            >>> loss = compute_loss(batch) * torch.from_numpy(weights)
        """
        buffer_size = self.num_valid

        # Get minimum priority for normalization (only from valid slots)
        p_min = float('inf')
        p_total = 0.0
        for i in range(self.capacity):
            if self.valid_mask[i]:
                p_min = min(p_min, self._it_min[i])
                p_total += self._it_sum[i]

        if p_total <= 0:
            # Fallback: uniform weights
            return np.ones(len(indices), dtype=np.float32)

        # Compute max weight for normalization
        # max_weight occurs at minimum priority
        max_weight = (p_min / p_total * buffer_size) ** (-beta)

        weights = []
        for idx in indices:
            # Get priority for this sample
            p_sample = self._it_sum[idx]
            # Compute probability
            prob = p_sample / p_total
            # Compute importance weight
            weight = (prob * buffer_size) ** (-beta)
            # Normalize by max weight
            weights.append(weight / max_weight)

        return np.array(weights, dtype=np.float32)

    def get_stats(self) -> dict:
        """Get comprehensive buffer statistics including priority, age, and freshness information."""
        base_stats = super().get_stats()

        # Fix: Use actual valid count for correct utilization calculation
        current_size = self.num_valid
        base_stats["current_size"] = current_size
        base_stats["utilization"] = current_size / self.capacity if self.capacity > 0 else 0.0
        base_stats["total_evicted"] = max(0, self.total_stored - current_size)  # Number of evicted samples
        base_stats["eviction_rate"] = base_stats["total_evicted"] / max(1, self.total_stored)

        # Add priority statistics from segment tree
        if current_size > 0:
            # Extract priorities from segment tree (only valid slots)
            priorities = []
            for i in range(self.capacity):
                if self.valid_mask[i]:
                    priorities.append(self._it_sum[i])
            priorities = np.array(priorities)
            base_stats.update({
                "priority/mean": float(priorities.mean()),
                "priority/std": float(priorities.std()),
                "priority/max": float(priorities.max()),
                "priority/min": float(priorities.min()),
                "priority_fn": self.priority_fn.__name__,
                "priority_exponent": self.priority_exponent,
                "max_priority": self._max_priority,
                # Age decay configuration - important for understanding sampling behavior
                "enable_age_decay": self.enable_age_decay,
                "age_decay": self.age_decay,
            })

            # Age distribution statistics (only for valid slots)
            ages = []
            for i in range(self.capacity):
                if self.valid_mask[i]:
                    ages.append(self.current_global_step - self.trajectories[i].global_step)
            ages = np.array(ages)
            base_stats.update({
                "age/mean": float(ages.mean()),
                "age/std": float(ages.std()),
                "age/max": float(ages.max()),  # Age of oldest policy
                "age/min": float(ages.min()),  # Age of newest policy
                "age/median": float(np.median(ages)),
                "age/p95": float(np.percentile(ages, 95)),  # 95th percentile age
            })

            # Estimate gradient step age (assuming constant replay ratio)
            estimated_replay_ratio = getattr(self, 'train_steps_per_env_step', 2.0)
            est_gradient_ages = ages * estimated_replay_ratio
            base_stats.update({
                "gradient_age/mean_est": float(est_gradient_ages.mean()),
                "gradient_age/max_est": float(est_gradient_ages.max()),
            })

            # Freshness metrics (based on age decay)
            freshness_weights = np.exp(-ages / self.age_decay)
            base_stats.update({
                "freshness/mean": float(freshness_weights.mean()),
                "freshness/std": float(freshness_weights.std()),
                "freshness/min": float(freshness_weights.min()),  # Least fresh (oldest)
                "freshness_ratio": float(np.sum(freshness_weights > 0.5) / current_size),  # Fraction with >50% freshness
            })

            # Sample count statistics (how often trajectories have been sampled)
            sample_counts = []
            for i in range(self.capacity):
                if self.valid_mask[i]:
                    sample_counts.append(self.trajectories[i].sample_count)
            sample_counts = np.array(sample_counts)
            if sample_counts.sum() > 0:
                base_stats.update({
                    "sample_count/mean": float(sample_counts.mean()),
                    "sample_count/max": float(sample_counts.max()),
                    "sample_count/never_sampled": float(np.sum(sample_counts == 0) / current_size),
                })

        return base_stats
