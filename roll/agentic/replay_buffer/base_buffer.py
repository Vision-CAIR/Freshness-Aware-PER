"""
Base Replay Buffer Interface for ROLL Framework

Provides a common interface for different types of replay buffers
to ensure consistent integration with the agentic pipeline.
"""

from abc import ABC, abstractmethod
from typing import Optional
from transformers import PreTrainedTokenizer
from roll.distributed.scheduler.protocol import DataProto


class BaseReplayBuffer(ABC):
    """
    Abstract base class for all replay buffers in the ROLL framework.
    
    This interface ensures that different replay buffer implementations
    (trajectory-level, step-level, etc.) can be used interchangeably
    in the agentic pipeline.
    """
    
    def __init__(self, capacity: int, batch_size: int, seed: int = 42):
        self.capacity = capacity
        self.batch_size = batch_size
        self.seed = seed
        self.total_stored = 0
        
    @abstractmethod
    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store data from a DataProto batch into the replay buffer.
        
        Args:
            batch: DataProto containing training data
            global_step: Current global training step
        """
        pass
    
    @abstractmethod
    def sample_for_training(self, batch_size: Optional[int] = None, device: str = 'cpu',
                            tokenizer: Optional[PreTrainedTokenizer] = None, sequence_length: int = 4096,
                            sampling_mode: str = "trajectory", steps_per_episode: int = 1,
                            sample_method: str = "uniform", candidates_per_group: int = 1,
                            group_sampling: str = "uniform") -> Optional[DataProto]:
        """
        Sample a batch of data for training.
        
        Args:
            batch_size: Optional override for default batch size
            device: Target device for tensors ('cpu' or 'cuda')
            tokenizer: Tokenizer for text processing
            sequence_length: Maximum sequence length for padding/truncation
            sampling_mode: "trajectory" or "step" sampling mode
            steps_per_episode: Number of steps per episode for step sampling
            sample_method: Sampling method ("uniform", "weighted", etc.)
            candidates_per_group: Number of candidates per group
            group_sampling: Group sampling strategy ("uniform", etc.)
            
        Returns:
            DataProto batch ready for training, or None if insufficient data
        """
        pass
    
    @abstractmethod
    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """
        Check if the buffer has enough data for sampling.
        
        Args:
            batch_size: Optional override for default batch size
            
        Returns:
            True if buffer can provide a batch of the requested size
        """
        pass
    
    @property
    @abstractmethod
    def buffer_type(self) -> str:
        """Return the type of buffer for identification."""
        pass
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        # Note: Subclasses should override this to provide accurate current_size
        return {
            "buffer_type": self.buffer_type,
            "capacity": self.capacity,
            "total_stored": self.total_stored,
            "utilization": self.total_stored / self.capacity if self.capacity > 0 else 0.0  # Will be overridden by subclasses
        }
