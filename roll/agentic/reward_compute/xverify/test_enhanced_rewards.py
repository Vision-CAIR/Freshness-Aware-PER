#!/usr/bin/env python3
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
Test script for enhanced reward functions with xverify and OTC GRPO support.

This script tests the integration of xverify evaluation and OTC GRPO rewards
for numina_math and xhpang_search datasets.

Usage:
    source /mnt/chensiheng/weiyu/env_activate.sh
    python test_enhanced_rewards.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl.utils.reward_score import _default_compute_score


def test_numina_math_enhanced():
    """Test enhanced numina_math reward with xverify and OTC"""
    print("=" * 60)
    print("Testing Enhanced Numina Math Reward")
    print("=" * 60)
    
    # Test case 1: Simple math problem with correct answer
    solution_1 = r"""
    Let me solve this step by step.
    
    We need to find the value of $\frac{1}{2} + \frac{1}{4}$.
    
    $\frac{1}{2} + \frac{1}{4} = \frac{2}{4} + \frac{1}{4} = \frac{3}{4}$
    
    Therefore, $\boxed{\frac{3}{4}}$
    """
    
    ground_truth_1 = "3/4"
    extra_info_1 = {
        "question": "What is the value of 1/2 + 1/4?",
        "query": "What is the value of 1/2 + 1/4?"
    }
    
    result_1 = _default_compute_score(
        data_source="numina_math",
        solution_str=solution_1,
        ground_truth=ground_truth_1,
        extra_info=extra_info_1
    )
    
    print(f"Test 1 - Simple math (correct): {result_1}")
    
    # Test case 2: Math problem with tool usage (should be penalized by OTC)
    solution_2 = r"""
    <search>fraction addition calculator</search>
    <search>1/2 + 1/4 calculation</search>
    <code>
    result = 0.5 + 0.25
    print(f"Result: {result}")
    </code>
    <execution_results>
    Result: 0.75
    </execution_results>
    
    Based on my calculations, $\frac{1}{2} + \frac{1}{4} = 0.75 = \frac{3}{4}$
    
    Therefore, $\boxed{\frac{3}{4}}$
    """
    
    result_2 = _default_compute_score(
        data_source="numina_math",
        solution_str=solution_2,
        ground_truth=ground_truth_1,
        extra_info=extra_info_1
    )
    
    print(f"Test 2 - Math with excessive tools: {result_2}")
    
    # Test case 3: Incorrect answer
    solution_3 = r"""
    $\frac{1}{2} + \frac{1}{4} = \frac{1}{6}$ (incorrect calculation)
    
    Therefore, $\boxed{\frac{1}{6}}$
    """
    
    result_3 = _default_compute_score(
        data_source="numina_math",
        solution_str=solution_3,
        ground_truth=ground_truth_1,
        extra_info=extra_info_1
    )
    
    print(f"Test 3 - Incorrect answer: {result_3}")


def test_xhpang_search_enhanced():
    """Test enhanced xhpang_search reward with xverify and OTC"""
    print("\n" + "=" * 60)
    print("Testing Enhanced XH Pang Search Reward")
    print("=" * 60)
    
    # Test case 1: Efficient search with correct answer
    solution_1 = """
    I need to find information about the capital of France.
    
    <search>capital of France</search>
    
    Based on my search results, the capital of France is Paris.
    
    <answer>Paris</answer>
    """
    
    ground_truth_1 = "Paris"
    extra_info_1 = {
        "question": "What is the capital of France?",
        "query": "What is the capital of France?"
    }
    
    result_1 = _default_compute_score(
        data_source="xhpang_search",
        solution_str=solution_1,
        ground_truth=ground_truth_1,
        extra_info=extra_info_1
    )
    
    print(f"Test 1 - Efficient search (correct): {result_1}")
    
    # Test case 2: Excessive search usage (should be penalized by OTC)
    solution_2 = """
    <search>capital of France</search>
    <search>France capital city</search>
    <search>Paris France capital</search>
    <search>what is capital of France</search>
    <code>
    print("Searching for France capital...")
    </code>
    <execution_results>
    Searching for France capital...
    </execution_results>
    
    After multiple searches, the capital of France is Paris.
    
    <answer>Paris</answer>
    """
    
    result_2 = _default_compute_score(
        data_source="xhpang_search",
        solution_str=solution_2,
        ground_truth=ground_truth_1,
        extra_info=extra_info_1
    )
    
    print(f"Test 2 - Excessive search tools: {result_2}")
    
    # Test case 3: Direct answer without search
    solution_3 = """
    The capital of France is Paris.
    
    <answer>Paris</answer>
    """
    
    result_3 = _default_compute_score(
        data_source="xhpang_search",
        solution_str=solution_3,
        ground_truth=ground_truth_1,
        extra_info=extra_info_1
    )
    
    print(f"Test 3 - Direct answer (no tools): {result_3}")
    
    # Test case 4: Multiple valid answers
    solution_4 = """
    <search>largest city in United States</search>
    
    The largest city in the United States is New York City.
    
    <answer>NYC</answer>
    """
    
    ground_truth_4 = {"target": ["New York City", "NYC", "New York"]}
    extra_info_4 = {
        "question": "What is the largest city in the United States?",
        "query": "What is the largest city in the United States?"
    }
    
    result_4 = _default_compute_score(
        data_source="xhpang_search",
        solution_str=solution_4,
        ground_truth=ground_truth_4,
        extra_info=extra_info_4
    )
    
    print(f"Test 4 - Multiple valid answers: {result_4}")


def test_fallback_modes():
    """Test fallback modes when xverify or OTC are not available"""
    print("\n" + "=" * 60)
    print("Testing Fallback Modes")
    print("=" * 60)
    
    # This will test the fallback behavior if xverify or OTC modules are not available
    try:
        # Import the reward modules directly to test fallback
        from verl.utils.reward_score import numina_math_reward, xhpang_search_reward
        
        # Test numina_math with xverify disabled
        result_math = numina_math_reward.compute_score(
            solution_str=r"The answer is $\boxed{2}$",
            ground_truth="2",
            question="What is 1+1?",
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )
        print(f"Numina Math (traditional mode): {result_math}")
        
        # Test xhpang_search with xverify disabled
        result_search = xhpang_search_reward.compute_score(
            solution_str="<answer>Paris</answer>",
            ground_truth="Paris",
            question="What is the capital of France?",
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )
        print(f"XH Pang Search (traditional mode): {result_search}")
        
    except Exception as e:
        print(f"Fallback test encountered error: {e}")


def main():
    """Main test function"""
    print("Enhanced Reward System Test Suite")
    print("=" * 60)
    print("Testing xverify and OTC GRPO integration")
    print("Environment should be activated with:")
    print("source /mnt/chensiheng/weiyu/env_activate.sh")
    print()
    
    try:
        test_numina_math_enhanced()
        test_xhpang_search_enhanced()
        test_fallback_modes()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 