"""
Utilities for implementing turn mode in old_prob computation.
"""

import torch
from typing import Dict, Tuple


def create_turn_mode_response_mask(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor = None,
    messages_list: list = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Create a modified response_mask for turn mode that only marks the last assistant turn.
    
    Args:
        response_mask: Original response mask marking all assistant responses [batch_size, seq_len]
        prompt_mask: Prompt mask if available [batch_size, seq_len]
        messages_list: List of messages for each sample in batch (optional, for debugging)
    
    Returns:
        step_response_mask: Modified mask only marking last assistant turn
        debug_info: Dictionary containing debug information
    """
    batch_size, seq_len = response_mask.shape
    step_response_mask = torch.zeros_like(response_mask)
    
    debug_info = {
        "num_turns": [],
        "last_turn_start": [],
        "last_turn_length": []
    }
    
    for batch_idx in range(batch_size):
        # Find all response segments in this sample
        response_segments = []
        in_response = False
        start_idx = 0
        
        for seq_idx in range(seq_len):
            if response_mask[batch_idx, seq_idx] == 1 and not in_response:
                # Start of a response segment
                in_response = True
                start_idx = seq_idx
            elif response_mask[batch_idx, seq_idx] == 0 and in_response:
                # End of a response segment
                in_response = False
                response_segments.append((start_idx, seq_idx))
        
        # Handle case where response extends to the end
        if in_response:
            response_segments.append((start_idx, seq_len))
        
        # Only mark the last response segment
        if response_segments:
            last_start, last_end = response_segments[-1]
            step_response_mask[batch_idx, last_start:last_end] = 1
            
            # Collect debug info
            debug_info["num_turns"].append(len(response_segments))
            debug_info["last_turn_start"].append(last_start)
            debug_info["last_turn_length"].append(last_end - last_start)
        else:
            # No response found (shouldn't happen in normal cases)
            debug_info["num_turns"].append(0)
            debug_info["last_turn_start"].append(-1)
            debug_info["last_turn_length"].append(0)
    
    # Convert debug info to tensors
    for key in debug_info:
        debug_info[key] = torch.tensor(debug_info[key])
    
    return step_response_mask, debug_info


def compute_step_mode_prompt_length(
    prompt_mask: torch.Tensor,
    step_response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the prompt length for step mode (everything before the last assistant response).
    
    Args:
        prompt_mask: Original prompt mask [batch_size, seq_len]
        step_response_mask: Step mode response mask [batch_size, seq_len]
    
    Returns:
        step_prompt_length: Length of prompt including all previous turns [batch_size]
    """
    batch_size = prompt_mask.shape[0]
    step_prompt_length = torch.zeros(batch_size, dtype=torch.long)
    
    for batch_idx in range(batch_size):
        # Find the start of the last response
        response_indices = torch.where(step_response_mask[batch_idx] == 1)[0]
        if len(response_indices) > 0:
            # Everything before the last response is considered prompt
            step_prompt_length[batch_idx] = response_indices[0]
        else:
            # If no response, entire sequence is prompt
            step_prompt_length[batch_idx] = prompt_mask[batch_idx].sum()
    
    return step_prompt_length
