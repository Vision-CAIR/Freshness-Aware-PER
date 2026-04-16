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

import re
import random
from typing import Dict, Any, Optional, Union, List
from .qa_em import extract_solution, normalize_answer, em_check

# Import xverify and OTC functionality
try:
    from .xverify.xverify_reward import xverify_evaluate_single, DEFAULT_XVERIFY_CONFIG
    XVERIFY_AVAILABLE = True
except ImportError:
    XVERIFY_AVAILABLE = False
    print("Warning: xverify not available, falling back to traditional EM evaluation")

try:
    from .otc_reward import compute_otc_reward, count_tool_calls
    OTC_AVAILABLE = True
except ImportError:
    OTC_AVAILABLE = False
    print("Warning: OTC reward not available, using base reward only")


def xverify_qa_evaluation(solution_str: str, 
                         ground_truth: Any, 
                         question: str = "",
                         model_config: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """
    Use xverify to evaluate QA answers.
    
    Args:
        solution_str: Complete solution string
        ground_truth: Ground truth answer (can be string, list, or dict)
        question: Original question (required for xverify)
        model_config: xverify model configuration
        
    Returns:
        Tuple of (is_correct, evaluation_response)
    """
    if not XVERIFY_AVAILABLE:
        # Fallback to traditional EM evaluation
        answer = extract_solution(solution_str=solution_str)
        is_correct = traditional_em_evaluation(answer, ground_truth)
        return is_correct, "Traditional EM evaluation (xverify unavailable)"
    
    if not question:
        # If no question provided, fall back to traditional method
        answer = extract_solution(solution_str=solution_str)
        is_correct = traditional_em_evaluation(answer, ground_truth)
        return is_correct, "Traditional EM evaluation (no question provided)"
    
    try:
        # Convert ground_truth to string format for xverify
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            gt_str = str(ground_truth['target'][0]) if isinstance(ground_truth['target'], list) else str(ground_truth['target'])
        elif isinstance(ground_truth, (list, tuple)):
            gt_str = str(ground_truth[0]) if ground_truth else ""
        else:
            gt_str = str(ground_truth)
        
        score, evaluation_response = xverify_evaluate_single(
            question, solution_str, gt_str, model_config
        )
        
        if score is None:
            # Error in xverify, fall back to traditional method
            answer = extract_solution(solution_str=solution_str)
            is_correct = traditional_em_evaluation(answer, ground_truth)
            return is_correct, f"Traditional EM evaluation (xverify error: {evaluation_response})"
        
        is_correct = (score == 2)  # xverify returns 2 for correct, 0 for incorrect
        return is_correct, evaluation_response
        
    except Exception as e:
        # Fall back to traditional method on any exception
        answer = extract_solution(solution_str=solution_str)
        is_correct = traditional_em_evaluation(answer, ground_truth)
        return is_correct, f"Traditional EM evaluation (xverify exception: {str(e)})"


def traditional_em_evaluation(answer: str, ground_truth: Any) -> bool:
    """
    Traditional exact match evaluation for QA tasks.
    
    Args:
        answer: Extracted answer from solution
        ground_truth: Ground truth answer in various formats
        
    Returns:
        Whether the answer is correct
    """
    if answer is None:
        return False
    
    # Handle different ground truth formats
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        return em_check(answer, ground_truth['target'])
    elif isinstance(ground_truth, (list, tuple)):
        return em_check(answer, ground_truth)
    else:
        # Handle comma-separated answers for multiple valid answers
        if isinstance(ground_truth, str) and ',' in ground_truth:
            candidate_answers = [ans.strip() for ans in ground_truth.split(',')]
            if len(candidate_answers) > 1 and all(ans for ans in candidate_answers):
                return em_check(answer, candidate_answers)
        return em_check(answer, ground_truth)


def compute_score(solution_str: str, 
                 ground_truth: Any, 
                 method: str = 'strict', 
                 format_score: float = 0., 
                 score: float = 1., 
                 return_dict: bool = False,
                 question: str = "",
                 extra_info: Optional[Dict[str, Any]] = None,
                 use_xverify: bool = True,
                 use_otc: bool = False,  # Conservative default
                 otc_method: str = "grpo",
                 correct_trajectories: Optional[List[str]] = None,
                 xverify_config: Optional[Dict[str, Any]] = None,
                 otc_alpha: float = 1.0,
                 otc_c: float = 1.0) -> Union[float, Dict[str, Any]]:
    """
    Enhanced XH Pang Search dataset reward function with xverify and OTC GRPO support.
    
    This function handles search-enhanced question answering tasks and combines:
    1. Traditional EM evaluation as fallback
    2. xverify-based evaluation for improved accuracy
    3. OTC GRPO reward to optimize tool usage (especially search patterns)
    
    Args:
        solution_str: The model's generated solution string
        ground_truth: The ground truth answer (can be string, list, or dict with 'target' key)
        method: Scoring method ('strict' or 'lenient')
        format_score: Score for format correctness when answer is wrong
        score: Score for correct answers
        return_dict: Whether to return detailed results as dict
        question: Original question (required for xverify evaluation)
        extra_info: Additional information (may contain question if not provided)
        use_xverify: Whether to use xverify for evaluation
        use_otc: Whether to apply OTC (Optimal Tool Call) reward
        otc_method: OTC method ("ppo" or "grpo")
        correct_trajectories: List of correct trajectories for OTC-GRPO
        xverify_config: Configuration for xverify model
        otc_alpha: Scale factor α for OTC tool reward integration
        otc_c: Smooth constant for OTC reward
        
    Returns:
        float or dict: The computed score or detailed results
    """
    
    # Handle case where question is passed via extra_info
    if not question and extra_info and 'question' in extra_info:
        question = extra_info['question']
    
    # Extract ground truth in proper format for consistent processing
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        # Keep original format for traditional EM check but extract string for debug
        gt_display = str(ground_truth['target'][0]) if isinstance(ground_truth['target'], list) else str(ground_truth['target'])
    elif isinstance(ground_truth, (list, tuple)):
        gt_display = str(ground_truth[0]) if ground_truth else ""
    else:
        gt_display = str(ground_truth)
    
    # 5% random sampling for debugging (reduced to minimize noise)
    do_print = random.random() < 0.05
    
    if do_print:
        print(f"======== XH PANG SEARCH REWARD DEBUG (5% sample) ========")
        print(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
        print(f"Ground truth (original): {str(ground_truth)[:100]}")
        print(f"Ground truth (processed): {gt_display}")
        print(f"Solution preview: {solution_str[:200]}...")
        print(f"Config: xverify={use_xverify}, otc={use_otc}")
        
        # Check for search and execution patterns
        try:
            search_count = len(re.findall(r'<search>.*?</search>', solution_str, re.DOTALL))
            code_count = len(re.findall(r'<code>.*?</code>', solution_str, re.DOTALL))
            execution_count = len(re.findall(r'<execution_results>.*?</execution_results>', solution_str, re.DOTALL))
            print(f"Patterns: search={search_count}, code={code_count}, exec={execution_count}")
            
            if use_otc and OTC_AVAILABLE:
                tool_count = count_tool_calls(solution_str)
                print(f"Tool calls detected: {tool_count}")
        except Exception as e:
            print(f"Pattern analysis error: {e}")

    # Evaluate answer correctness
    if use_xverify and XVERIFY_AVAILABLE:
        is_correct, evaluation_response = xverify_qa_evaluation(
            solution_str, ground_truth, question, xverify_config or DEFAULT_XVERIFY_CONFIG
        )
        evaluation_method = "xverify"
    else:
        # Traditional EM evaluation
        answer = extract_solution(solution_str=solution_str)
        is_correct = traditional_em_evaluation(answer, ground_truth)
        evaluation_response = f"Traditional EM evaluation (answer: {answer})"
        evaluation_method = "traditional_em"
    
    # Base score calculation
    base_score = score if is_correct else format_score
    
    # Apply OTC reward if enabled
    if use_otc and OTC_AVAILABLE:
        # Validate OTC method and requirements
        if otc_method.lower() == "grpo" and not correct_trajectories:
            if do_print:
                print("Warning: GRPO method requires correct_trajectories, falling back to base score")
            final_score = base_score
            otc_info = {"error": "GRPO requires correct_trajectories"}
        elif otc_method.lower() == "ppo":
            # OTC-PPO doesn't require correct_trajectories, proceed normally
            def base_reward_fn(sol_str, gt):
                # Return the base score we already calculated
                return base_score
            
            try:
                otc_result = compute_otc_reward(
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    method="ppo",  # Explicitly set to PPO
                    base_reward_fn=base_reward_fn,
                    correct_trajectories=None,  # PPO doesn't need this
                    c=otc_c,
                    alpha=otc_alpha,
                    return_dict=True
                )
                final_score = otc_result["score"]
                otc_info = otc_result
            except Exception as e:
                if do_print:
                    print(f"OTC-PPO reward error: {e}, using base score")
                final_score = base_score
                otc_info = {"error": str(e), "fallback_reason": "otc_ppo_computation_failed"}
        else:
            # OTC-GRPO with correct_trajectories provided
            def base_reward_fn(sol_str, gt):
                # Return the base score we already calculated
                return base_score
            
            try:
                otc_result = compute_otc_reward(
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    method="grpo",  # Explicitly set to GRPO
                    base_reward_fn=base_reward_fn,
                    correct_trajectories=correct_trajectories,
                    c=otc_c,
                    alpha=otc_alpha,
                    return_dict=True
                )
                final_score = otc_result["score"]
                otc_info = otc_result
            except Exception as e:
                if do_print:
                    print(f"OTC-GRPO reward error: {e}, using base score")
                final_score = base_score
                otc_info = {"error": str(e), "fallback_reason": "otc_grpo_computation_failed"}
    else:
        final_score = base_score
        if use_otc and not OTC_AVAILABLE:
            otc_info = {"error": "OTC not available", "fallback_reason": "otc_import_failed"}
        else:
            otc_info = {"used": False}

    if do_print:
        print(f"Evaluation: {evaluation_method} | Correct: {is_correct}")
        print(f"Scores: base={base_score:.3f}, final={final_score:.3f}")
        print(f"Response: {evaluation_response[:100]}...")
        if otc_info:
            print(f"OTC: {otc_info}")
        print(f"=========================================================")
    
    # Return dict format if requested
    if return_dict:
        result = {
            "score": final_score,
            "base_score": base_score,
            "is_correct": is_correct,
            "evaluation_method": evaluation_method,
            "evaluation_response": evaluation_response,
            "extracted_answer": extract_solution(solution_str=solution_str) if not use_xverify else "N/A (xverify)",
            "has_search_pattern": bool(re.search(r'<search>.*?</search>', solution_str, re.DOTALL)),
            "has_execution_pattern": bool(re.search(r'<execution_results>.*?</execution_results>', solution_str, re.DOTALL)),
            "use_xverify": use_xverify,
            "use_otc": use_otc,
            "data_source": "xhpang_search"  # Add data source identifier
        }
        
        # Add OTC information if available
        if otc_info:
            result["otc_info"] = otc_info
            
        return result
    
    return final_score


def compute_score_em(solution_str: str, 
                    ground_truth: Any, 
                    method: str = 'strict', 
                    format_score: float = 0., 
                    score: float = 1., 
                    return_dict: bool = False) -> Union[float, Dict[str, Any]]:
    """
    Wrapper function for compatibility with existing EM interface.
    """
    return compute_score(solution_str, ground_truth, method, format_score, score, return_dict, use_xverify=False, use_otc=False)


if __name__ == '__main__':
    # Test cases for enhanced XH Pang Search dataset
    print("=== Testing Enhanced XH Pang Search Reward Function ===")
    
    # Test case 1: Search-enhanced correct answer with xverify
    test_solution_1 = """I need to find information about the capital of France.

<search>capital of France</search>

Based on my search, I can see that Paris is the capital of France.

<answer>Paris</answer>"""
    
    test_gt_1 = "Paris"
    test_question_1 = "What is the capital of France?"
    score_1 = compute_score(test_solution_1, test_gt_1, question=test_question_1, return_dict=True, use_xverify=True)
    print(f"Test 1 - Search enhanced with xverify: {score_1}")
    
    # Test case 2: Direct answer without search, traditional evaluation
    test_solution_2 = """The capital of France is Paris.

<answer>Paris</answer>"""
    
    score_2 = compute_score(test_solution_2, test_gt_1, return_dict=True, use_xverify=False, use_otc=False)
    print(f"Test 2 - Direct answer, traditional: {score_2}")
    
    # Test case 3: With OTC-PPO (no correct_trajectories needed)
    test_solution_3 = """<search>France capital</search>
<search>Paris France</search>
<code>print("Paris is the capital")</code>
<answer>Paris</answer>"""
    
    score_3 = compute_score(test_solution_3, test_gt_1, question=test_question_1, return_dict=True, use_otc=True, otc_method="ppo")
    print(f"Test 3 - Multiple tools with OTC-PPO: {score_3}")
    
    # Test case 4: With OTC-GRPO (requires correct_trajectories)
    test_solution_4 = """<search>France capital</search>
<search>Paris France</search>
<code>print("Paris is the capital")</code>
<answer>Paris</answer>"""
    
    correct_trajectories = ["<search>capital France</search>\nParis", "Paris"]  # Optimal uses 1 tool
    score_4 = compute_score(test_solution_4, test_gt_1, question=test_question_1, return_dict=True, 
                           use_otc=True, otc_method="grpo", correct_trajectories=correct_trajectories)
    print(f"Test 4 - Multiple tools with OTC-GRPO: {score_4}")
    
    # Test case 5: Wrong answer with OTC-PPO (should be 0)
    test_solution_5 = """<search>France capital</search>
<answer>London</answer>"""  # Wrong answer
    
    score_5 = compute_score(test_solution_5, test_gt_1, question=test_question_1, return_dict=True, use_otc=True, otc_method="ppo")
    print(f"Test 5 - Wrong answer with OTC-PPO: {score_5}")
    
    # Test case 6: Multiple valid answers
    test_solution_6 = """<answer>NYC</answer>"""
    test_gt_6 = {"target": ["New York City", "NYC", "New York"]}
    test_question_6 = "What is the largest city in the US?"
    
    score_6 = compute_score(test_solution_6, test_gt_6, question=test_question_6, return_dict=True)
    print(f"Test 6 - Multiple valid answers: {score_6}")
    
    print("=== Enhanced XH Pang Search Tests Complete ===\n")