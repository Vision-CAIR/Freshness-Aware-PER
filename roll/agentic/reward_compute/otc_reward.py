# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OTC (Optimal Tool Call) Reward Implementation

This module implements the OTC-GRPO reward mechanism from the paper:
r_φ^tool(q,y) = α · r_tool · r_φ(q,y)

Where:
- r_φ(q,y): Base reward function (accuracy)  
- r_tool: Tool efficiency reward
- α: Scale factor for tool reward integration
"""

import re
import math
import random
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict


def count_tool_calls(solution_str: str, tool_patterns: Optional[List[str]] = None) -> int:
    """
    Count the number of tool calls in a solution string.
    
    Args:
        solution_str: The solution text to analyze
        tool_patterns: List of regex patterns to match tool calls
                      If None, uses default patterns for common tools
    
    Returns:
        Number of tool calls found
    """
    if tool_patterns is None:
        # Default patterns for common tool usage
        tool_patterns = [
            r'<code>.*?</code>',           # Code execution blocks
            r'<search>.*?</search>',       # Search queries
            r'web_search\s*\(',            # Web search function calls
            r'search_r1\s*\(',             # Search R1 function calls
            r'calculator\s*\(',            # Calculator function calls
            r'python\s*\(',                # Python function calls
            r'tool_call\s*\(',             # Generic tool calls
        ]
    
    total_calls = 0
    for pattern in tool_patterns:
        matches = re.findall(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        total_calls += len(matches)
    
    return total_calls


def compute_mapping_function(m: int, n: int) -> float:
    """
    Compute the mapping function f(m, n) for OTC-GRPO.
    
    Formula:
    f(m, n) = {
        0,           if m = 0 and n = 0
        m,           if n = 0  
        2nm/(m+n),   otherwise (harmonic mean variant)
    }
    
    Args:
        m: Current trajectory tool call count
        n: Estimated optimal tool call count
        
    Returns:
        Mapped value
    """
    if m == 0 and n == 0:
        return 0.0
    elif n == 0:
        return float(m)
    else:
        # Harmonic mean variant: 2nm/(m+n)
        denominator = m + n
        if denominator == 0:
            return 0.0
        return 2.0 * n * m / denominator


def compute_tool_reward(m: int, n: int, c: float = 1.0) -> float:
    """
    Compute tool efficiency reward r_tool based on OTC-GRPO formula.
    
    Formula:
    r_tool = {
        1,                           if f(m,n) = n = 0
        cos(m * π / (2m + c)),      if n = 0
        sin(f(m,n) * π / (2n)),     otherwise
    }
    
    Args:
        m: Number of tool calls in current trajectory
        n: Estimated optimal number of tool calls
        c: Smooth constant for cosine reward when n=0
        
    Returns:
        Tool usage reward (0 to 1)
    """
    f_mn = compute_mapping_function(m, n)
    
    # Case 1: f(m,n) = n = 0 (optimal solution requires no tools and none used)
    if f_mn == n and n == 0:
        return 1.0
    
    # Case 2: n = 0 (optimal solution requires no tools but tools were used)
    elif n == 0:
        if m == 0:
            return 1.0  # Perfect score for no tool usage
        
        # Compute the cosine reward
        denominator = 2 * m + c
        if denominator == 0:
            return 0.0
        
        angle = m * math.pi / denominator
        reward = math.cos(angle)
        return max(0.0, reward)
    
    # Case 3: General case (optimal solution requires tools)
    else:
        if n == 0:
            return 0.0  # Avoid division by zero
        
        angle = f_mn * math.pi / (2.0 * n)
        reward = math.sin(angle)
        
        # Ensure reward is in [0, 1] range
        return max(0.0, reward)


def find_optimal_tool_calls(correct_trajectories: List[str], 
                          tool_patterns: Optional[List[str]] = None) -> int:
    """
    Find the optimal (minimum) number of tool calls from correct trajectories.
    
    This is used in OTC-GRPO to estimate the optimal tool call count.
    
    Args:
        correct_trajectories: List of solution strings that led to correct answers
        tool_patterns: Patterns to match tool calls
        
    Returns:
        Minimum number of tool calls among correct trajectories
    """
    if not correct_trajectories:
        return 0
    
    tool_counts = []
    for trajectory in correct_trajectories:
        count = count_tool_calls(trajectory, tool_patterns)
        tool_counts.append(count)
    
    return min(tool_counts)


def compute_otc_reward(solution_str: str, 
                      ground_truth: Any,
                      method: str = "grpo",
                      base_reward_fn: Optional[callable] = None,
                      correct_trajectories: Optional[List[str]] = None,
                      tool_patterns: Optional[List[str]] = None,
                      c: float = 1.0,
                      alpha: float = 1.0,
                      return_dict: bool = False) -> Union[float, Dict[str, Any]]:
    """
    Compute OTC reward using the standard paper formula:
    r_φ^tool(q,y) = α · r_tool · r_φ(q,y)
    
    Args:
        solution_str: The solution text to evaluate
        ground_truth: Ground truth for base reward calculation
        method: "ppo" or "grpo" (for GRPO, need correct_trajectories)
        base_reward_fn: Function to compute base reward r_φ(q,y)
                       If None, assumes the answer is correct (reward=1.0)
        correct_trajectories: List of correct solution strings (needed for GRPO)
        tool_patterns: Patterns to match tool calls
        c: Smooth constant for cosine reward
        alpha: Scale factor α for tool reward integration
        return_dict: Whether to return detailed information
        
    Returns:
        Combined reward or dictionary with detailed information
    """
    # 5% random debug printing (reduced to minimize noise)
    do_print = random.random() < 0.05
    
    # Count tool calls in current solution
    m = count_tool_calls(solution_str, tool_patterns)
    
    # Compute base reward r_φ(q,y)
    if base_reward_fn is not None:
        base_reward = base_reward_fn(solution_str, ground_truth)
        if isinstance(base_reward, dict):
            base_score = base_reward.get("score", 0.0)
        else:
            base_score = float(base_reward)
    else:
        # Assume correct answer if no base reward function provided
        base_score = 1.0
    
    # Estimate optimal tool calls n
    if method.lower() == "grpo" and correct_trajectories is not None:
        n = find_optimal_tool_calls(correct_trajectories, tool_patterns)
    else:
        # For PPO or when no group information available, assume n=0 (no tools needed)
        n = 0
    
    # Compute tool efficiency reward r_tool
    tool_reward = compute_tool_reward(m, n, c)
    
    # Apply paper formula: r_φ^tool(q,y) = α · r_tool · r_φ(q,y)
    final_reward = alpha * tool_reward * base_score
    
    if do_print:
        print(f"======== OTC REWARD DEBUG (5% sample) ========")
        print(f"Method: {method.upper()} | Tools: m={m}, n={n}")
        print(f"Rewards: base={base_score:.4f}, tool={tool_reward:.4f}")
        print(f"Final: {alpha}*{tool_reward:.4f}*{base_score:.4f} = {final_reward:.4f}")
        print(f"GT: {str(ground_truth)[:50]}...")
        print(f"Sol: {solution_str[:150]}...")
        print(f"===============================================")
    
    if return_dict:
        result = {
            "score": final_reward,
            "base_score": base_score,
            "tool_reward": tool_reward,
            "tool_calls": m,
            "optimal_tool_calls": n,
            "method": method.lower(),
            "alpha": alpha,
            "mapping_function_value": compute_mapping_function(m, n)
        }
        return result
    
    return final_reward


# Convenience functions for backward compatibility
def compute_score_otc_ppo(solution_str: str, 
                         ground_truth: Any,
                         base_reward_fn: Optional[callable] = None,
                         c: float = 1.0,
                         alpha: float = 1.0) -> Dict[str, Any]:
    """Convenience function for OTC-PPO reward calculation."""
    return compute_otc_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method="ppo",
        base_reward_fn=base_reward_fn,
        c=c,
        alpha=alpha,
        return_dict=True
    )


def compute_score_otc_grpo(solution_str: str,
                          ground_truth: Any,
                          correct_trajectories: List[str],
                          base_reward_fn: Optional[callable] = None,
                          c: float = 1.0,
                          alpha: float = 1.0) -> Dict[str, Any]:
    """Convenience function for OTC-GRPO reward calculation."""
    return compute_otc_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method="grpo",
        base_reward_fn=base_reward_fn,
        correct_trajectories=correct_trajectories,
        c=c,
        alpha=alpha,
        return_dict=True
    )


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Clean OTC Reward Function ===")
    
    # Test solution with tools
    test_solution = """
    I need to search for information first.
    <search>What is the capital of France?</search>
    
    Let me also calculate something.
    <code>
    result = 2 + 2
    print(result)
    </code>
    
    The answer is Paris.
    """
    
    # Test cases
    def correct_answer_fn(sol_str, gt):
        return 1.0  # Correct answer
    
    def wrong_answer_fn(sol_str, gt):
        return 0.0  # Wrong answer
    
    # Test correct answer with tools
    result_correct = compute_otc_reward(
        solution_str=test_solution,
        ground_truth="Paris",
        method="grpo",
        base_reward_fn=correct_answer_fn,
        correct_trajectories=["<search>capital France</search>\nParis"],  # Optimal uses 1 tool
        return_dict=True
    )
    print(f"Correct answer with tools: {result_correct}")
    
    # Test wrong answer with tools (should be 0)
    result_wrong = compute_otc_reward(
        solution_str=test_solution,
        ground_truth="Paris",
        method="grpo", 
        base_reward_fn=wrong_answer_fn,
        correct_trajectories=["<search>capital France</search>\nParis"],
        return_dict=True
    )
    print(f"Wrong answer with tools: {result_wrong}")
    
    print("=== Clean OTC Tests Complete ===") 