"""
Step-Level Replay Buffer for ROLL Framework

Specifically designed to work with StepEnvManager data format.
Stores individual conversation steps with step-level rewards and penalties.
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
class StepEntry:
    """
    Single step entry for step-level replay buffer.
    Matches the output format of StepEnvManager.
    """
    # Core data from DataProto
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    response_mask: np.ndarray
    prompt_mask: np.ndarray
    scores: np.ndarray  # Token-level scores with step reward on response tokens
    penalty: float      # Step-level penalty scalar
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
    state_hash: str  # Added missing state_hash field
    step: int  # CRITICAL: Step index within episode, required for gigpo

    # Episode termination signals (following Stable-Baselines3/Tianshou convention)
    done: bool = False          # True if this is the last step of the episode
    terminated: bool = False    # True if episode ended due to environment termination
    truncated: bool = False     # True if episode ended due to time limit

    # Storage metadata
    stored_at_step: int = 0
    step_length: int = 0

    # Priority-related metadata
    priority: float = 1.0       # Current priority value (intrinsic value)
    sample_count: int = 0       # Number of times sampled (for statistics)
    global_step: int = 0        # Global training step when stored (for age calculation)


class StepReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer specialized for step-level data from StepEnvManager.
    
    Key Design Principles:
    1. Step-level storage: individual conversation turns as atomic units
    2. StepEnvManager compatibility: matches exact data format
    3. Step-level penalty handling: penalties are per-step values, not cumulative
    4. Efficient step sampling: can sample from any step independently
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        batch_size: int = 128,
        seed: int = 42,
        priority_fn: callable = None,
        priority_exponent: float = 0.6,
        enable_nstep: bool = False,
        n_step: int = 5,
        gamma: float = 0.99,
        enable_age_decay: bool = False,
        age_decay: float = 1000.0,
    ):
        super().__init__(capacity, batch_size, seed)
        self.steps = deque(maxlen=capacity)
        self.rng = random.Random(seed)

        # Priority configuration
        from .priority_functions import uniform_priority
        self.priority_fn = priority_fn or uniform_priority
        self.priority_exponent = priority_exponent

        # Segment Tree for O(log n) prioritized sampling
        self._tree_capacity = next_power_of_2(capacity)
        self._it_sum = SumSegmentTree(self._tree_capacity)
        self._it_min = MinSegmentTree(self._tree_capacity)
        self._max_priority = 1.0

        # N-Step configuration
        self.enable_nstep = enable_nstep
        self.n_step = n_step
        self.gamma = gamma

        # Age-based freshness weighting (optional, default off for standard PER behavior)
        self.enable_age_decay = enable_age_decay
        self.age_decay = age_decay
        self.current_global_step = 0

        # Episode Index for n-step returns
        self._episode_index: Dict[str, Dict[int, int]] = {}
        self._buffer_to_episode: Dict[int, Tuple[str, int]] = {}

        logger.info(
            f"StepReplayBuffer: capacity={capacity}, priority_fn={self.priority_fn.__name__}, "
            f"enable_nstep={enable_nstep}, n_step={n_step}, gamma={gamma}, "
            f"enable_age_decay={enable_age_decay}, age_decay={age_decay}"
        )
    
    @property
    def buffer_type(self) -> str:
        return "step"

    @property
    def num_valid(self) -> int:
        """Return current number of valid steps in buffer."""
        return len(self.steps)

    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store step data from StepEnvManager.

        Args:
            batch: DataProto from StepEnvManager containing individual steps
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
                # CRITICAL FIX: If no behavior_log_probs, create zeros with length input_ids - 1
                # to align with next-token prediction semantics
                behavior_log_probs = np.zeros_like(input_ids[:-1], dtype=np.float32)
            
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
            state_hash = batch.non_tensor_batch["state_hash"][i]  # Extract state_hash
            
            # Extract step index - CRITICAL for gigpo algorithm
            step = int(batch.non_tensor_batch["step"][i]) if "step" in batch.non_tensor_batch else 0

            # Extract episode termination signals (following Stable-Baselines3/Tianshou convention)
            done = bool(batch.non_tensor_batch["done"][i]) if "done" in batch.non_tensor_batch else False
            terminated = bool(batch.non_tensor_batch["terminated"][i]) if "terminated" in batch.non_tensor_batch else False
            truncated = bool(batch.non_tensor_batch["truncated"][i]) if "truncated" in batch.non_tensor_batch else False

            # Calculate step length from attention mask
            step_length = int(attention_mask.sum())

            step_entry = StepEntry(
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
                state_hash=state_hash,  # Add state_hash
                step=step,  # Add step index for gigpo
                done=done,  # Episode termination flag
                terminated=terminated,  # Environment termination flag
                truncated=truncated,  # Time limit truncation flag
                stored_at_step=global_step,
                step_length=step_length,
                global_step=global_step  # Store global step for age calculation
            )

            # Calculate priority for this step
            # Pass age_decay as kwarg for reward_fresh and other age-aware priority functions
            try:
                priority = self.priority_fn(step_entry, global_step, age_decay=self.age_decay)
                step_entry.priority = float(priority)
            except Exception as e:
                logger.warning(f"Failed to calculate priority, using default 1.0: {e}")
                step_entry.priority = 1.0

            # IMPORTANT: Use total_stored for consistent indexing across buffer wrap-around
            # deque automatically handles capacity, but segment tree needs explicit index
            current_idx = self.total_stored % self.capacity

            # Handle eviction when buffer is full (clean up episode index)
            if len(self.steps) == self.capacity:
                self._cleanup_evicted_step(current_idx)

            # Update segment trees with max_priority^alpha (standard PER convention)
            # New samples get max_priority to ensure they're sampled at least once
            # The actual priority will be updated after training via update_priorities()
            priority_alpha = self._max_priority ** self.priority_exponent
            self._it_sum[current_idx] = priority_alpha
            self._it_min[current_idx] = priority_alpha

            # Update episode index for n-step returns
            if traj_id not in self._episode_index:
                self._episode_index[traj_id] = {}
            self._episode_index[traj_id][step] = current_idx
            self._buffer_to_episode[current_idx] = (traj_id, step)

            # Append to deque (auto-evicts oldest when full)
            self.steps.append(step_entry)
            self.total_stored += 1

        # Periodic garbage collection to prevent memory leaks
        if self.total_stored % 100000 == 0 and self.total_stored > 0:
            import gc
            gc.collect()
            logger.info(f"Replay buffer GC triggered at {self.total_stored} steps stored")

        # Debug: Count done flags in this batch
        done_count_in_batch = sum(1 for i in range(batch_size) if "done" in batch.non_tensor_batch and bool(batch.non_tensor_batch["done"][i]))

        logger.info(
            f"[STORE] Stored {batch_size} steps. Total: {len(self.steps)}, "
            f"episode_index_size={len(self._episode_index)}, "
            f"buffer_to_episode_size={len(self._buffer_to_episode)}, "
            f"done_count_in_batch={done_count_in_batch}"
        )
    
    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough steps for sampling."""
        required_size = batch_size or self.batch_size
        return len(self.steps) >= required_size
    
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
        Sample steps and reconstruct DataProto format.

        Args:
            batch_size: Number of steps to sample
            device: Target device for tensors ('cpu' or 'cuda')
            tokenizer: Tokenizer for text processing
            sequence_length: Maximum sequence length for padding/truncation
            sampling_mode: "trajectory" or "step" sampling mode (ignored, always step)
            steps_per_episode: Number of steps per episode (ignored for step mode)
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
            logger.debug(f"Insufficient steps for sampling: {len(self.steps)} < {sample_size}")
            return None, []

        # Sample steps based on priority function
        # Deterministic strategies: uniform, lifo, fifo
        # Weighted strategies: reward, td_error, recency, combined, etc.
        buffer_list = list(self.steps)
        buffer_size = len(buffer_list)
        priority_fn_name = self.priority_fn.__name__

        # Track sampled indices for priority updates and importance weights
        sampled_indices = []

        if priority_fn_name == "lifo_priority":
            # LIFO (Last-In-First-Out): Deterministic sampling of newest N steps
            # Recommended for Echo mode (train_steps_per_env_step=1) for near-on-policy training
            start_idx = max(0, buffer_size - sample_size)
            sampled_indices = list(range(start_idx, buffer_size))
            sampled_steps = buffer_list[start_idx:]
            logger.debug(f"LIFO sampling: selected last {len(sampled_steps)} steps (indices {start_idx} to {buffer_size})")

        elif priority_fn_name == "fifo_priority":
            # FIFO (First-In-First-Out): Deterministic sampling of oldest N steps
            # Ensures all data is used before eviction
            sampled_indices = list(range(sample_size))
            sampled_steps = buffer_list[:sample_size]
            logger.debug(f"FIFO sampling: selected first {len(sampled_steps)} steps")

        elif priority_fn_name == "uniform_priority":
            # Uniform random sampling: Standard replay buffer behavior
            sampled_indices = self.rng.sample(range(buffer_size), sample_size)
            sampled_steps = [buffer_list[i] for i in sampled_indices]
            logger.debug(f"Uniform sampling: randomly selected {len(sampled_steps)} steps")

        else:
            # Weighted priority-based sampling using Segment Tree (O(log n) per sample)
            # This is the core of Prioritized Experience Replay (PER)
            sampled_indices = self._sample_proportional(sample_size, buffer_size)
            sampled_steps = [buffer_list[i] for i in sampled_indices]

            # Update sample counts for statistics
            for idx in sampled_indices:
                buffer_list[idx].sample_count += 1

            logger.debug(f"PER sampling ({priority_fn_name}): sampled {len(sampled_steps)} steps with priority_alpha={self.priority_exponent}")
        
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
        state_hashes = []  # Add state_hashes list
        steps = []  # Add steps list for gigpo
        
        for i, step in enumerate(sampled_steps):
            original_seq_len = len(step.input_ids)
            
            # Apply consistent truncation/padding like RolloutScheduler
            effective_seq_len = min(original_seq_len, max_seq_len)
            
            # Use pad_to_length for consistent behavior with original ROLL env_manager
            step_input_ids = torch.from_numpy(step.input_ids)
            step_attention_mask = torch.from_numpy(step.attention_mask.astype(bool))
            step_position_ids = torch.from_numpy(step.position_ids)
            step_response_mask = torch.from_numpy(step.response_mask.astype(bool))
            step_prompt_mask = torch.from_numpy(step.prompt_mask.astype(bool))
            step_scores = torch.from_numpy(step.scores)
            
            batch_input_ids[i] = pad_to_length(step_input_ids, max_seq_len, pad_token_id)
            # Keep attention_mask padded with 0 to match RolloutScheduler
            batch_attention_mask[i] = pad_to_length(step_attention_mask, max_seq_len, 0)
            batch_position_ids[i] = pad_to_length(step_position_ids, max_seq_len, 0)
            batch_response_mask[i] = pad_to_length(step_response_mask, max_seq_len, 0)
            batch_prompt_mask[i] = pad_to_length(step_prompt_mask, max_seq_len, 0)
            batch_scores[i] = pad_to_length(step_scores, max_seq_len, 0.0)
            
            batch_penalties[i] = step.penalty
            
            # Handle behavior_log_probs with pad_to_length to next-token length (sequence_length-1)
            step_behavior_log_probs = torch.from_numpy(step.behavior_log_probs)
            batch_old_log_probs[i] = pad_to_length(step_behavior_log_probs, max_seq_len - 1, 0.0)
            
            # Collect non-tensor data
            env_ids.append(step.env_id)
            group_ids.append(step.group_id)
            messages_lists.append(step.messages_list)
            tags.append(step.tag)
            frames_lists.append(step.frames)
            step_scores_lists.append(step.step_scores)
            episode_scores_lists.append(step.episode_scores)
            traj_group_ids.append(step.traj_group_id)
            traj_ids.append(step.traj_id)
            state_hashes.append(step.state_hash)  # Collect state_hash
            steps.append(step.step)  # Collect step index for gigpo
        
        # Create DataProto in the exact format StepEnvManager produces
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
            "state_hash": np.array(state_hashes, dtype=object),  # Add state_hash field
            "step": np.array(steps, dtype=object),  # CRITICAL: Add step field for gigpo
        }
        
        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "step",
            "sample_size": sample_size,
            "buffer_utilization": len(self.steps) / self.capacity,
            "sampled_indices": sampled_indices  # For priority update after training
        }

        # Compute importance weights for PER (off-policy correction)
        if compute_importance_weights and priority_fn_name not in ["lifo_priority", "fifo_priority", "uniform_priority"]:
            importance_weights = self.compute_importance_weights(sampled_indices, beta=importance_weight_beta)
            dataproto.batch["importance_weights"] = torch.from_numpy(importance_weights).to(target_device)
            logger.debug(f"Computed importance weights with beta={importance_weight_beta:.2f}, mean={importance_weights.mean():.4f}")

        # Compute n-step returns if enabled
        if self.enable_nstep:
            nstep_returns, completeness_mask = self.compute_nstep_returns(
                sampled_indices=sampled_indices,
                n_step=self.n_step,
                gamma=self.gamma,
                bootstrap_values=None  # Bootstrap values computed in pipeline if needed
            )
            dataproto.batch["nstep_returns"] = torch.from_numpy(nstep_returns).to(target_device)
            dataproto.batch["nstep_completeness"] = torch.from_numpy(completeness_mask.astype(np.float32)).to(target_device)
            dataproto.meta_info["nstep_complete_ratio"] = float(completeness_mask.mean())
            logger.debug(
                f"Computed n-step returns: n_step={self.n_step}, gamma={self.gamma}, "
                f"complete_ratio={completeness_mask.mean():.2f}"
            )

        logger.debug(f"Sampled {sample_size} steps for training (indices: {len(sampled_indices)})")
        return dataproto, sampled_indices

    def sample_episodes_for_hierarchical(
        self,
        num_episodes: int,
        steps_per_episode: int = None,  # Now optional - if None, use actual episode lengths
        device: str = 'cpu',
        tokenizer: Optional[PreTrainedTokenizer] = None,
        sequence_length: int = 4096,
        compute_importance_weights: bool = False,
        importance_weight_beta: float = 0.4,
        max_retries: int = 100,
        min_episode_length: int = 1  # Minimum episode length to accept
    ) -> Optional[Tuple[DataProto, List[List[int]]]]:
        """
        Sample complete episodes for Hierarchical RL.

        This method samples COMPLETE episodes (from step 0 to done=True),
        respecting natural episode boundaries. Episodes can have variable lengths.

        Following Stable-Baselines3/Tianshou convention:
        - Uses `done` flag to identify episode boundaries
        - Supports variable-length episodes (no fixed steps_per_episode requirement)
        - Returns `done` mask for GAE computation

        Args:
            num_episodes: Number of episodes to sample
            steps_per_episode: If provided, used as max_steps hint. If None, use actual lengths.
            device: Target device for tensors
            tokenizer: Tokenizer for padding
            sequence_length: Maximum sequence length for padding
            compute_importance_weights: Whether to compute PER importance weights
            importance_weight_beta: Beta parameter for importance weights
            max_retries: Maximum retries (unused in new implementation)
            min_episode_length: Minimum episode length to accept (default 1)

        Returns:
            Tuple of (DataProto, episode_boundaries):
            - DataProto: Batch containing all steps from sampled episodes, with `done` mask
            - episode_boundaries: List of starting indices for each episode in the batch
        """
        if len(self._episode_index) == 0:
            logger.warning("[HIER_SAMPLE] No episodes in buffer - episode_index is empty")
            return None, []

        buffer_list = list(self.steps)
        priority_fn_name = self.priority_fn.__name__

        # Build list of complete episodes from the index
        # An episode is "complete" if it has step 0 and at least one step with done=True
        complete_episodes = []
        for traj_id, step_dict in self._episode_index.items():
            if 0 not in step_dict:
                continue  # Skip incomplete episodes (no step 0)

            # Get all steps for this episode in order
            sorted_steps = sorted(step_dict.keys())
            indices = [step_dict[s] for s in sorted_steps]
            episode_length = len(indices)

            # Check minimum length requirement
            if episode_length < min_episode_length:
                continue

            # Verify the last step has done=True (episode actually ended)
            last_step_entry = buffer_list[indices[-1]]
            if not last_step_entry.done:
                continue  # Episode not yet complete

            complete_episodes.append({
                'traj_id': traj_id,
                'indices': indices,
                'length': episode_length
            })

        # Log episode statistics
        if complete_episodes:
            ep_lengths = [ep['length'] for ep in complete_episodes]
            logger.info(
                f"[HIER_SAMPLE] Found {len(complete_episodes)} complete episodes "
                f"(total in index: {len(self._episode_index)}). "
                f"Length stats: min={min(ep_lengths)}, max={max(ep_lengths)}, "
                f"mean={sum(ep_lengths)/len(ep_lengths):.1f}"
            )
        else:
            # Detailed diagnostics - check why episodes are not complete
            has_step_0 = sum(1 for ep in self._episode_index.values() if 0 in ep)
            total_steps = sum(len(ep) for ep in self._episode_index.values())

            # Debug: Check done flags in buffer
            done_count = sum(1 for entry in buffer_list if entry.done)

            # Debug: Sample a few episodes to see their structure
            sample_debug_info = []
            for traj_id, step_dict in list(self._episode_index.items())[:3]:
                sorted_steps = sorted(step_dict.keys())
                indices = [step_dict[s] for s in sorted_steps]
                if indices:
                    # Check if index is valid
                    last_idx = indices[-1]
                    if last_idx < len(buffer_list):
                        last_entry = buffer_list[last_idx]
                        sample_debug_info.append(
                            f"traj={traj_id[:8]}, steps={sorted_steps}, "
                            f"last_idx={last_idx}, done={last_entry.done}, step_attr={last_entry.step}"
                        )
                    else:
                        sample_debug_info.append(
                            f"traj={traj_id[:8]}, steps={sorted_steps}, "
                            f"last_idx={last_idx} OUT OF RANGE (buffer_len={len(buffer_list)})"
                        )

            logger.warning(
                f"[HIER_SAMPLE] No complete episodes found! "
                f"Episodes with step 0: {has_step_0}/{len(self._episode_index)}, "
                f"total indexed steps: {total_steps}, "
                f"min_episode_length required: {min_episode_length}, "
                f"done_count in buffer: {done_count}/{len(buffer_list)}"
            )
            if sample_debug_info:
                logger.warning(f"[HIER_SAMPLE] Sample episodes: {sample_debug_info}")
            return None, []

        # Sample episodes
        if len(complete_episodes) >= num_episodes:
            # Enough episodes - sample without replacement
            if priority_fn_name in ["lifo_priority", "fifo_priority", "uniform_priority"]:
                sampled_episodes = self.rng.sample(complete_episodes, num_episodes)
            else:
                # Priority-based: compute episode priority as mean of step priorities
                episode_priorities = []
                for ep in complete_episodes:
                    ep_priority = sum(buffer_list[idx].priority for idx in ep['indices']) / len(ep['indices'])
                    episode_priorities.append(max(ep_priority, 1e-8))

                total_priority = sum(episode_priorities)
                probs = [p / total_priority for p in episode_priorities]
                sampled_indices = self.rng.choices(range(len(complete_episodes)), weights=probs, k=num_episodes)
                sampled_episodes = [complete_episodes[i] for i in sampled_indices]
        else:
            # Not enough episodes - sample with replacement
            logger.info(
                f"[HIER_SAMPLE] Only {len(complete_episodes)} complete episodes available, "
                f"sampling {num_episodes} with replacement"
            )
            sampled_episodes = self.rng.choices(complete_episodes, k=num_episodes)

        # Build the batch - flatten episodes while tracking boundaries
        all_indices = []
        episode_boundaries = []
        episode_lengths = []

        for ep in sampled_episodes:
            episode_boundaries.append(len(all_indices))
            episode_lengths.append(len(ep['indices']))
            all_indices.extend(ep['indices'])

            # Update sample counts
            if priority_fn_name not in ["lifo_priority", "fifo_priority", "uniform_priority"]:
                for idx in ep['indices']:
                    buffer_list[idx].sample_count += 1

        logger.info(
            f"[HIER_SAMPLE] Sampled {len(sampled_episodes)} episodes, "
            f"total steps: {len(all_indices)}, "
            f"episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, "
            f"mean={sum(episode_lengths)/len(episode_lengths):.1f}"
        )

        total_samples = len(all_indices)
        max_seq_len = sequence_length
        target_device = torch.device(device if device is not None else 'cpu')

        # Prepare tensors
        batch_input_ids = torch.zeros((total_samples, max_seq_len), dtype=torch.long, device=target_device)
        batch_attention_mask = torch.zeros((total_samples, max_seq_len), dtype=torch.bool, device=target_device)
        batch_position_ids = torch.zeros((total_samples, max_seq_len), dtype=torch.long, device=target_device)
        batch_response_mask = torch.zeros((total_samples, max_seq_len), dtype=torch.bool, device=target_device)
        batch_prompt_mask = torch.zeros((total_samples, max_seq_len), dtype=torch.bool, device=target_device)
        batch_scores = torch.zeros((total_samples, max_seq_len), dtype=torch.float, device=target_device)
        batch_penalties = torch.zeros(total_samples, dtype=torch.float, device=target_device)
        batch_old_log_probs = torch.zeros((total_samples, max_seq_len - 1), dtype=torch.float, device=target_device)

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
        state_hashes = []
        steps = []
        dones = []  # Episode termination flags (for GAE computation)

        # Fill batch from sampled indices
        for i, buffer_idx in enumerate(all_indices):
            step_entry = buffer_list[buffer_idx]

            # Convert numpy arrays to tensors
            step_input_ids = torch.from_numpy(step_entry.input_ids)
            step_attention_mask = torch.from_numpy(step_entry.attention_mask.astype(bool))
            step_position_ids = torch.from_numpy(step_entry.position_ids)
            step_response_mask = torch.from_numpy(step_entry.response_mask.astype(bool))
            step_prompt_mask = torch.from_numpy(step_entry.prompt_mask.astype(bool))
            step_scores = torch.from_numpy(step_entry.scores)
            step_behavior_log_probs = torch.from_numpy(step_entry.behavior_log_probs)

            # Pad tensors
            batch_input_ids[i] = pad_to_length(step_input_ids, max_seq_len, pad_token_id)
            batch_attention_mask[i] = pad_to_length(step_attention_mask, max_seq_len, 0)
            batch_position_ids[i] = pad_to_length(step_position_ids, max_seq_len, 0)
            batch_response_mask[i] = pad_to_length(step_response_mask, max_seq_len, 0)
            batch_prompt_mask[i] = pad_to_length(step_prompt_mask, max_seq_len, 0)
            batch_scores[i] = pad_to_length(step_scores, max_seq_len, 0.0)
            batch_penalties[i] = step_entry.penalty
            batch_old_log_probs[i] = pad_to_length(step_behavior_log_probs, max_seq_len - 1, 0.0)

            # Collect non-tensor data
            env_ids.append(step_entry.env_id)
            group_ids.append(step_entry.group_id)
            messages_lists.append(step_entry.messages_list)
            tags.append(step_entry.tag)
            frames_lists.append(step_entry.frames)
            step_scores_lists.append(step_entry.step_scores)
            episode_scores_lists.append(step_entry.episode_scores)
            traj_group_ids.append(step_entry.traj_group_id)
            traj_ids.append(step_entry.traj_id)
            state_hashes.append(step_entry.state_hash)
            steps.append(step_entry.step)
            dones.append(step_entry.done)  # Collect done flags

        # Create done mask tensor (CRITICAL for GAE computation)
        # Following Stable-Baselines3 convention: done[t] = 1 if step t is terminal
        batch_dones = torch.tensor(dones, dtype=torch.float, device=target_device)

        # Create DataProto
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
            "dones": batch_dones,  # Episode termination flags for GAE
        }, batch_size=[total_samples])

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
            "state_hash": np.array(state_hashes, dtype=object),
            "step": np.array(steps, dtype=object),
            "done": np.array(dones, dtype=object),  # Also in non_tensor for compatibility
        }

        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "step",
            "sampling_mode": "hierarchical",
            "num_episodes": len(sampled_episodes),
            "episode_lengths": episode_lengths,  # Variable lengths per episode
            "episode_boundaries": episode_boundaries,  # CRITICAL: Episode structure for hierarchical RL
            "total_samples": total_samples,
            "buffer_utilization": len(self.steps) / self.capacity,
            "sampled_indices": all_indices,
            "dones": batch_dones,  # Episode termination flags for GAE (tensor)
        }

        # Compute importance weights if requested
        if compute_importance_weights and priority_fn_name not in ["lifo_priority", "fifo_priority", "uniform_priority"]:
            importance_weights = self.compute_importance_weights(all_indices, beta=importance_weight_beta)
            dataproto.batch["importance_weights"] = torch.from_numpy(importance_weights).to(target_device)
            logger.debug(f"Computed importance weights with beta={importance_weight_beta:.2f}, mean={importance_weights.mean():.4f}")

        return dataproto, episode_boundaries

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
        Update priorities for sampled steps after training (standard PER).

        Standard PER formula:
            tree[idx] = priority^α
            max_priority = max(max_priority, priority)

        Optional age decay (when enable_age_decay=True):
            effective_priority = priority × exp(-age / age_decay)
            tree[idx] = effective_priority^α

        Time complexity: O(k * log n) where k = len(indices)

        Args:
            indices: Buffer indices of sampled steps
            priorities: New priority values (e.g., |advantage|, |TD-error|)
            current_global_step: Current training step for age calculation (only used if enable_age_decay=True)

        Example:
            >>> # After training and computing advantages
            >>> advantages = batch["advantages"]  # [batch_size, seq_len]
            >>> response_mask = batch["response_mask"]  # [batch_size, seq_len]
            >>> priorities = masked_mean(abs(advantages), response_mask).cpu().numpy()
            >>> buffer.update_priorities(sampled_indices, priorities, current_global_step)
        """
        assert len(indices) == len(priorities), \
            f"Indices and priorities length mismatch: {len(indices)} vs {len(priorities)}"

        buffer_list = list(self.steps)
        global_step = current_global_step if current_global_step is not None else self.current_global_step

        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < len(buffer_list)):
                logger.warning(f"Invalid index {idx} for buffer size {len(buffer_list)}, skipping")
                continue

            # Ensure priority is positive (add small epsilon)
            priority = max(float(priority), 1e-6)

            # Update step entry priority (store raw value for debugging)
            buffer_list[idx].priority = priority

            # Compute effective priority (with optional age decay)
            if self.enable_age_decay:
                sample_age = global_step - buffer_list[idx].global_step
                freshness_weight = np.exp(-sample_age / self.age_decay)
                effective_priority = priority * freshness_weight
            else:
                effective_priority = priority

            # Update segment trees with priority^alpha (standard PER)
            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[idx] = priority_alpha
            self._it_min[idx] = priority_alpha

            # Track maximum priority
            self._max_priority = max(self._max_priority, priority)

        logger.debug(f"Updated priorities for {len(indices)} steps. "
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
            current_global_step: Current training step (optional)

        Returns:
            Effective priority value
        """
        buffer_list = list(self.steps)
        if not (0 <= idx < len(buffer_list)):
            return 0.0

        if self.enable_age_decay:
            global_step = current_global_step if current_global_step is not None else self.current_global_step
            sample_age = global_step - buffer_list[idx].global_step
            freshness_weight = np.exp(-sample_age / self.age_decay)
            return buffer_list[idx].priority * freshness_weight
        else:
            return buffer_list[idx].priority

    def refresh_all_age_decay(self, current_global_step: int) -> int:
        """
        Refresh age decay for ALL samples in the buffer, updating segment tree.

        This method should be called periodically (e.g., every training step) to ensure
        that old samples have their priorities properly decayed based on their age.

        Without this refresh, only sampled entries get their age decay updated via
        update_priorities(), leaving unsampled old entries with stale (too high) priorities.

        Time complexity: O(buffer_size)
        Typical runtime: ~50-100ms for buffer_size=100,000

        This method is designed to be called asynchronously during GPU training,
        so the CPU overhead is hidden by GPU computation time.

        Args:
            current_global_step: Current training step for age calculation

        Returns:
            Number of samples refreshed

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

        buffer_list = list(self.steps)
        buffer_size = len(buffer_list)

        if buffer_size == 0:
            return 0

        refreshed_count = 0
        max_age = 0
        total_age = 0

        for idx, entry in enumerate(buffer_list):
            if entry is None:
                continue

            # Calculate current age and freshness
            age = max(0, current_global_step - entry.global_step)
            freshness_weight = np.exp(-age / self.age_decay)

            # Calculate effective priority with age decay
            effective_priority = max(entry.priority * freshness_weight, 1e-8)

            # Update segment tree
            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[idx] = priority_alpha
            self._it_min[idx] = priority_alpha

            refreshed_count += 1
            max_age = max(max_age, age)
            total_age += age

        avg_age = total_age / refreshed_count if refreshed_count > 0 else 0

        logger.debug(
            f"[AGE_DECAY] Refreshed {refreshed_count} samples at step {current_global_step}. "
            f"avg_age={avg_age:.1f}, max_age={max_age}, age_decay={self.age_decay}"
        )

        return refreshed_count

    @staticmethod
    def compute_advantage_priorities(batch: DataProto) -> np.ndarray:
        """
        Compute advantage-based priorities from a DataProto batch.

        This extracts advantages from the batch and computes mean absolute advantage
        per trajectory/step, masked by response tokens.

        Args:
            batch: DataProto containing "advantages" and "response_mask"

        Returns:
            priorities: [batch_size] array of advantage-based priorities

        Example:
            >>> # After compute_advantage() in pipeline
            >>> priorities = StepReplayBuffer.compute_advantage_priorities(batch)
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
            indices: Buffer indices of sampled steps
            beta: Importance weight exponent (0 = no correction, 1 = full correction)

        Returns:
            Importance weights normalized by max weight, shape [batch_size]

        Example:
            >>> indices, weights = buffer.sample_with_weights(batch_size=128, beta=0.6)
            >>> loss = compute_loss(batch) * torch.from_numpy(weights)
        """
        buffer_size = len(self.steps)

        # Get minimum priority for normalization
        p_min = self._it_min.min(0, buffer_size)
        p_total = self._it_sum.sum(0, buffer_size)

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
        """Get buffer statistics including priority information."""
        base_stats = super().get_stats()

        # Fix: Use actual buffer size for correct utilization calculation
        current_size = len(self.steps)
        base_stats["current_size"] = current_size
        base_stats["utilization"] = current_size / self.capacity if self.capacity > 0 else 0.0
        base_stats["total_evicted"] = max(0, self.total_stored - current_size)  # Number of evicted samples

        # Add priority statistics from segment tree
        if current_size > 0:
            # Extract priorities from segment tree
            priorities = np.array([self._it_sum[i] for i in range(current_size)])
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

        # Add episode index statistics
        if self._episode_index:
            base_stats.update({
                "episode_index/num_episodes": len(self._episode_index),
                "episode_index/avg_episode_length": np.mean([len(ep) for ep in self._episode_index.values()]),
                "episode_index/total_indexed_steps": sum(len(ep) for ep in self._episode_index.values()),
            })

        return base_stats

    def _cleanup_evicted_step(self, evicted_idx: int) -> None:
        """
        Clean up episode index when a step is evicted from buffer.

        Args:
            evicted_idx: Buffer index being evicted (overwritten)
        """
        if evicted_idx not in self._buffer_to_episode:
            return

        old_traj_id, old_step = self._buffer_to_episode[evicted_idx]

        # Remove from episode index
        if old_traj_id in self._episode_index:
            if old_step in self._episode_index[old_traj_id]:
                del self._episode_index[old_traj_id][old_step]

            # If episode is now empty, remove it entirely
            if not self._episode_index[old_traj_id]:
                del self._episode_index[old_traj_id]
                logger.debug(f"Episode {old_traj_id} fully evicted from buffer")

        # Remove from reverse mapping
        del self._buffer_to_episode[evicted_idx]

    def get_nstep_indices(
        self,
        start_idx: int,
        n_step: int
    ) -> Tuple[List[int], bool]:
        """
        Get next n steps from the same episode (tianshou-style traversal).

        This is the core method for n-step returns and GAE computation.
        It mimics tianshou's next() method but uses explicit episode index.

        Args:
            start_idx: Starting buffer index
            n_step: Number of steps to look ahead

        Returns:
            (indices, complete):
            - indices: List of buffer indices [start_idx, next_idx, ...], length <= n_step
            - complete: True if we got full n steps without hitting episode boundary

        Example:
            >>> indices, complete = buffer.get_nstep_indices(42, n_step=5)
            >>> if complete:
            >>>     # We have 5 consecutive steps: indices = [42, 43, 44, 45, 46]
            >>> else:
            >>>     # Episode ended early: indices = [42, 43] (only 2 steps available)
        """
        if start_idx not in self._buffer_to_episode:
            logger.warning(f"Buffer index {start_idx} not in episode index")
            return [start_idx], False

        traj_id, start_step = self._buffer_to_episode[start_idx]

        if traj_id not in self._episode_index:
            logger.warning(f"Episode {traj_id} not in episode index")
            return [start_idx], False

        episode = self._episode_index[traj_id]
        max_step = max(episode.keys())  # Last step in this episode

        indices = []
        for offset in range(n_step):
            target_step = start_step + offset

            # Check if step exists in episode
            if target_step not in episode:
                # Hit episode boundary (step was evicted or doesn't exist)
                return indices, False

            indices.append(episode[target_step])

            # Check if this is the last step (after adding it to indices)
            if target_step == max_step:
                # Reached end of episode, but we've collected up to max_step
                # Return False because we can't get more steps beyond this
                return indices, False

        # Got full n steps without hitting boundary
        return indices, True

    def build_stacked_indices(
        self,
        sampled_indices: List[int],
        n_step: int
    ) -> np.ndarray:
        """
        Build stacked indices like tianshou: [n_step, batch_size].

        This enables vectorized n-step return computation.

        Args:
            sampled_indices: Starting buffer indices [batch_size]
            n_step: Number of steps to stack

        Returns:
            stacked_indices: [n_step, batch_size]
            Each column contains n consecutive steps from same episode

        Example:
            >>> sampled_indices = [42, 100, 200]  # batch_size=3
            >>> stacked = buffer.build_stacked_indices(sampled_indices, n_step=5)
            >>> # stacked shape: [5, 3]
            >>> # stacked[:, 0] = [42, 43, 44, 45, 46]  # 5 steps from episode starting at 42
            >>> # stacked[:, 1] = [100, 101, 101, 101, 101]  # Only 2 steps, then repeats last
            >>> # stacked[:, 2] = [200, 201, 202, 203, 204]  # Full 5 steps
        """
        batch_size = len(sampled_indices)
        stacked = np.zeros((n_step, batch_size), dtype=np.int32)

        for i, start_idx in enumerate(sampled_indices):
            indices, complete = self.get_nstep_indices(start_idx, n_step)

            # Fill stacked array
            for step in range(n_step):
                if step < len(indices):
                    stacked[step, i] = indices[step]
                else:
                    # Episode ended early - repeat last index (tianshou convention)
                    # This allows vectorized computation while handling variable episode lengths
                    stacked[step, i] = indices[-1] if indices else start_idx

        return stacked

    def compute_nstep_returns(
        self,
        sampled_indices: List[int],
        n_step: Optional[int] = None,
        gamma: Optional[float] = None,
        bootstrap_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute n-step returns for sampled steps.

        N-step return formula:
        R_t^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})

        Args:
            sampled_indices: Starting buffer indices [batch_size]
            n_step: Number of steps for return computation (default: self.n_step)
            gamma: Discount factor (default: self.gamma)
            bootstrap_values: Optional bootstrapped values [batch_size]
                            If None, no bootstrap is added (Monte Carlo return)

        Returns:
            Tuple of:
            - nstep_returns: [batch_size] computed n-step returns
            - completeness_mask: [batch_size] boolean array
                                True if full n-step sequence was available
                                False if hit episode boundary early

        Example:
            >>> returns, complete = buffer.compute_nstep_returns(
            ...     sampled_indices=[0, 10, 20],
            ...     n_step=5,
            ...     gamma=0.99,
            ...     bootstrap_values=critic_values
            ... )
            >>> # returns.shape = [3], complete.shape = [3]
            >>> # complete[0] = True means indices[0] had full 5-step sequence
        """
        n_step = n_step or self.n_step
        gamma = gamma or self.gamma

        batch_size = len(sampled_indices)
        returns = np.zeros(batch_size, dtype=np.float32)
        completeness_mask = np.zeros(batch_size, dtype=bool)

        buffer_list = list(self.steps)

        for i, start_idx in enumerate(sampled_indices):
            # Get n-step trajectory
            indices, complete = self.get_nstep_indices(start_idx, n_step)
            completeness_mask[i] = complete

            if not indices:
                logger.warning(f"No indices found for start_idx={start_idx}, traj may be evicted")
                continue

            # Accumulate discounted rewards
            discount = 1.0
            for idx in indices:
                if idx >= len(buffer_list):
                    logger.warning(f"Index {idx} out of bounds for buffer size {len(buffer_list)}")
                    break

                step_entry = buffer_list[idx]

                # Extract step reward: sum over response tokens
                response_mask_bool = step_entry.response_mask.astype(bool)
                reward = float(step_entry.scores[response_mask_bool].sum())

                returns[i] += discount * reward
                discount *= gamma

            # Add bootstrap value if trajectory is complete and values provided
            if complete and bootstrap_values is not None:
                returns[i] += discount * bootstrap_values[i]

        return returns, completeness_mask

    def compute_gae(
        self,
        sampled_indices: List[int],
        values: np.ndarray,
        gamma: Optional[float] = None,
        lambda_: float = 0.95,
        n_step: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) for sampled steps.

        GAE formula:
        A_t = Σ_{k=0}^∞ (γλ)^k * δ_{t+k}
        where: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)

        This implementation uses truncated GAE with horizon n_step.

        Args:
            sampled_indices: Starting buffer indices [batch_size]
            values: State values V(s_t) for each sampled step [batch_size]
            gamma: Discount factor (default: self.gamma)
            lambda_: GAE lambda parameter (exponential smoothing)
            n_step: Truncation horizon for GAE computation

        Returns:
            Tuple of:
            - advantages: [batch_size] computed GAE advantages
            - completeness_mask: [batch_size] boolean array indicating complete sequences

        Example:
            >>> advantages, complete = buffer.compute_gae(
            ...     sampled_indices=[0, 10, 20],
            ...     values=critic_values,  # [3]
            ...     gamma=0.99,
            ...     lambda_=0.95,
            ...     n_step=20
            ... )
        """
        gamma = gamma or self.gamma

        batch_size = len(sampled_indices)
        advantages = np.zeros(batch_size, dtype=np.float32)
        completeness_mask = np.zeros(batch_size, dtype=bool)

        buffer_list = list(self.steps)

        for i, start_idx in enumerate(sampled_indices):
            # Get trajectory for GAE computation
            indices, complete = self.get_nstep_indices(start_idx, n_step)
            completeness_mask[i] = complete

            if len(indices) < 2:
                # Need at least 2 steps to compute TD error
                advantages[i] = 0.0
                continue

            # Compute TD errors for each step in the trajectory
            deltas = []
            for j, idx in enumerate(indices[:-1]):  # Exclude last step
                if idx >= len(buffer_list):
                    break

                step_entry = buffer_list[idx]
                next_idx = indices[j + 1]

                if next_idx >= len(buffer_list):
                    break

                next_step_entry = buffer_list[next_idx]

                # Extract reward
                response_mask_bool = step_entry.response_mask.astype(bool)
                reward = float(step_entry.scores[response_mask_bool].sum())

                # Get values
                # For the first step, use provided value; for others, reuse if available
                # Note: This is a simplified version. Ideally, we should recompute values
                # for all steps, but that would require a critic forward pass
                v_t = values[i] if j == 0 else 0.0  # Simplified
                v_next = 0.0  # Simplified

                # Compute TD error
                delta = reward + gamma * v_next - v_t
                deltas.append(delta)

            # Compute GAE using backward accumulation
            gae = 0.0
            for delta in reversed(deltas):
                gae = delta + gamma * lambda_ * gae

            advantages[i] = gae

        return advantages, completeness_mask