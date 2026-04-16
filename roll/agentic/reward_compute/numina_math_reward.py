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

import random
import re
import math
from fractions import Fraction
from typing import Dict, Any, Optional, Union, List
from .qa_em import extract_solution
from .prime_math import grade_answer as prime_math_grade_answer

# Import xverify and OTC functionality
try:
    from .xverify.xverify_reward import xverify_evaluate_single, DEFAULT_XVERIFY_CONFIG
    XVERIFY_AVAILABLE = True
except ImportError:
    XVERIFY_AVAILABLE = False
    print("Warning: xverify not available, falling back to traditional math grading")

try:
    from .otc_reward import compute_otc_reward, count_tool_calls
    OTC_AVAILABLE = True
except ImportError:
    OTC_AVAILABLE = False
    print("Warning: OTC reward not available, using base reward only")


def normalize_basic(expr):
    """Basic normalization for mathematical expressions"""
    if expr is None:
        return ""
    
    # Convert to string and strip whitespace
    expr = str(expr).strip()
    
    # Remove common formatting
    expr = expr.replace(" ", "").replace("\\,", "").replace(",", "")
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    expr = expr.replace("\\frac", "frac").replace("\\tfrac", "frac")
    
    # Normalize fractions format
    expr = re.sub(r'frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)
    
    # Normalize common mathematical constants
    expr = expr.replace("\\pi", "pi").replace("π", "pi")
    expr = expr.replace("\\sqrt", "sqrt")
    
    return expr.lower()


def quick_string_match(answer, ground_truth):
    """First layer: Quick string-based matching"""
    if answer is None or ground_truth is None:
        return False
    
    # Direct string match
    if str(answer).strip() == str(ground_truth).strip():
        return True
    
    # Normalized string match
    norm_answer = normalize_basic(answer)
    norm_gt = normalize_basic(ground_truth)
    
    if norm_answer == norm_gt:
        return True
    
    # Remove common variations
    variations = [
        (r'[()]', ''),  # Remove parentheses
        (r'\.0+$', ''),  # Remove trailing zeros
        (r'^0+', ''),   # Remove leading zeros
    ]
    
    for pattern, replacement in variations:
        norm_answer_var = re.sub(pattern, replacement, norm_answer)
        norm_gt_var = re.sub(pattern, replacement, norm_gt)
        if norm_answer_var == norm_gt_var:
            return True
    
    return False


def safe_numeric_comparison(answer, ground_truth, tolerance=1e-9):
    """Second layer: Safe numerical comparison"""
    try:
        # Try direct float conversion
        try:
            num_answer = float(answer)
            num_gt = float(ground_truth)
            return abs(num_answer - num_gt) < tolerance
        except (ValueError, TypeError):
            pass
        
        # Try fraction conversion
        try:
            # Handle common fraction formats
            answer_clean = str(answer).replace(" ", "").replace("\\frac", "").replace("{", "").replace("}", "")
            gt_clean = str(ground_truth).replace(" ", "").replace("\\frac", "").replace("{", "").replace("}", "")
            
            # Parse fractions like "1/2" or "(1)/(2)"
            answer_clean = re.sub(r'\(([^)]+)\)/\(([^)]+)\)', r'\1/\2', answer_clean)
            gt_clean = re.sub(r'\(([^)]+)\)/\(([^)]+)\)', r'\1/\2', gt_clean)
            
            frac_answer = Fraction(answer_clean)
            frac_gt = Fraction(gt_clean)
            return frac_answer == frac_gt
        except (ValueError, ZeroDivisionError):
            pass
        
        # Try percentage conversion
        if '%' in str(answer) or '%' in str(ground_truth):
            try:
                answer_pct = float(str(answer).replace('%', '')) / 100 if '%' in str(answer) else float(answer)
                gt_pct = float(str(ground_truth).replace('%', '')) / 100 if '%' in str(ground_truth) else float(ground_truth)
                return abs(answer_pct - gt_pct) < tolerance
            except (ValueError, TypeError):
                pass
                
    except Exception:
        pass
    
    return False


def is_simple_expression(expr):
    """Check if expression is simple enough for SymPy processing"""
    if expr is None:
        return False
    
    expr_str = str(expr)
    
    # Length check
    if len(expr_str) > 50:
        return False
    
    # Check for dangerous patterns that can cause memory issues
    dangerous_patterns = [
        r'\^.*\^',           # Nested exponents
        r'2\^\{.*\}',        # Large exponentials
        r'log.*log.*log',    # Nested logarithms
        r'\d{10,}',          # Very large numbers
        r'\^[1-9]\d{2,}',    # High exponents (>99)
        r'\^\{[1-9]\d{2,}\}', # High exponents in braces (>99)
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expr_str):
            return False
    
    # Only allow simple operations
    allowed_chars = set('0123456789+-*/()^.abcdefghijklmnopqrstuvwxyz \\{}[]')
    if not all(c.lower() in allowed_chars for c in expr_str):
        return False
    
    return True


def limited_sympy_check(answer, ground_truth, timeout_seconds=3):
    """Third layer: Limited SymPy check with strict timeout"""
    try:
        # Additional safety check before calling SymPy
        if not is_simple_expression(answer) or not is_simple_expression(ground_truth):
            return False
        
        # Use original prime_math logic but with more restrictive conditions
        # We'll only call it if both expressions are deemed safe
        return prime_math_grade_answer(answer, ground_truth)
        
    except Exception:
        # Any exception in SymPy means we fall back to False
        return False


def hybrid_math_grading(answer, ground_truth):
    """
    Hybrid strategy for mathematical expression comparison:
    1. Quick string matching (fastest, safest)
    2. Safe numerical comparison (fast, handles numbers and fractions)
    3. Limited SymPy check (slower, only for simple expressions)
    """
    if answer is None or ground_truth is None:
        return False
    
    # Layer 1: Quick string matching
    if quick_string_match(answer, ground_truth):
        return True
    
    # Layer 2: Safe numerical comparison  
    if safe_numeric_comparison(answer, ground_truth):
        return True
    
    # Layer 3: Try prime_math grading directly (more lenient than limited SymPy)
    try:
        return prime_math_grade_answer(answer, ground_truth)
    except Exception:
        # If all else fails, do a final simple comparison
        try:
            # Simple normalization and comparison
            norm_answer = str(answer).strip().lower().replace(" ", "")
            norm_gt = str(ground_truth).strip().lower().replace(" ", "")
            return norm_answer == norm_gt
        except Exception:
            return False


def xverify_math_evaluation(solution_str: str, 
                          ground_truth: Any, 
                          question: str = "",
                          model_config: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """
    Use xverify to evaluate mathematical answers.
    
    Args:
        solution_str: Complete solution string
        ground_truth: Ground truth answer  
        question: Original question (required for xverify)
        model_config: xverify model configuration
        
    Returns:
        Tuple of (is_correct, evaluation_response)
    """
    if not XVERIFY_AVAILABLE:
        # Fallback to traditional evaluation
        answer = extract_solution(solution_str=solution_str)
        is_correct = hybrid_math_grading(answer, ground_truth)
        return is_correct, "Traditional math grading (xverify unavailable)"
    
    if not question:
        # If no question provided, fall back to traditional method
        answer = extract_solution(solution_str=solution_str)
        is_correct = hybrid_math_grading(answer, ground_truth)
        return is_correct, "Traditional math grading (no question provided)"
    
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
            is_correct = hybrid_math_grading(answer, gt_str)
            return is_correct, f"Traditional math grading (xverify error: {evaluation_response})"
        
        is_correct = (score == 2)  # xverify returns 2 for correct, 0 for incorrect
        return is_correct, evaluation_response
        
    except Exception as e:
        # Fall back to traditional method on any exception
        answer = extract_solution(solution_str=solution_str)
        # Extract ground truth properly for fallback
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            gt_str = str(ground_truth['target'][0]) if isinstance(ground_truth['target'], list) else str(ground_truth['target'])
        elif isinstance(ground_truth, (list, tuple)):
            gt_str = str(ground_truth[0]) if ground_truth else ""
        else:
            gt_str = str(ground_truth)
        is_correct = hybrid_math_grading(answer, gt_str)
        return is_correct, f"Traditional math grading (xverify exception: {str(e)})"


def compute_score(solution_str: str, 
                 ground_truth: Any, 
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
    Enhanced Numina Math dataset reward function with xverify and OTC GRPO support.
    
    This function combines:
    1. Traditional math grading as fallback
    2. xverify-based evaluation for improved accuracy
    3. OTC GRPO reward to optimize tool usage
    
    Args:
        solution_str: The model's generated solution string.
        ground_truth: The ground truth mathematical answer.
        format_score: Score for format correctness when answer is wrong.
        score: Score for correct answers.
        return_dict: Whether to return detailed results as a dict.
        question: Original question (required for xverify evaluation).
        extra_info: Additional information (may contain question if not provided).
        use_xverify: Whether to use xverify for evaluation.
        use_otc: Whether to apply OTC (Optimal Tool Call) reward.
        otc_method: OTC method ("ppo" or "grpo").
        correct_trajectories: List of correct trajectories for OTC-GRPO.
        xverify_config: Configuration for xverify model.
        otc_alpha: Scale factor α for OTC tool reward integration.
        otc_c: Smooth constant for OTC reward.
        
    Returns:
        float or dict: The computed score or detailed results.
    """
    
    # Handle case where question is passed via extra_info
    if not question and extra_info and 'question' in extra_info:
        question = extra_info['question']
    
    # Extract ground truth in proper format
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        gt_str = str(ground_truth['target'][0]) if isinstance(ground_truth['target'], list) else str(ground_truth['target'])
    elif isinstance(ground_truth, (list, tuple)):
        gt_str = str(ground_truth[0]) if ground_truth else ""
    else:
        gt_str = str(ground_truth)
    
    # 5% random sampling for debugging (reduced to minimize noise)
    do_print = random.random() < 0.05
    
    if do_print:
        print(f"======== NUMINA MATH REWARD DEBUG (5% sample) ========")
        print(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
        print(f"Ground truth (original): {str(ground_truth)[:100]}")
        print(f"Ground truth (processed): {gt_str}")
        print(f"Solution preview: {solution_str[:200]}...")
        print(f"Config: xverify={use_xverify}, otc={use_otc}")
        if use_otc and OTC_AVAILABLE:
            try:
                tool_count = count_tool_calls(solution_str)
                print(f"Tool calls detected: {tool_count}")
            except Exception as e:
                print(f"Tool count error: {e}")

    # Evaluate answer correctness
    if use_xverify and XVERIFY_AVAILABLE:
        is_correct, evaluation_response = xverify_math_evaluation(
            solution_str, ground_truth, question, xverify_config or DEFAULT_XVERIFY_CONFIG
        )
        evaluation_method = "xverify"
    else:
        # Traditional evaluation
        answer = extract_solution(solution_str=solution_str)
        if answer is None:
            is_correct = False
            evaluation_response = "No answer extracted"
        else:
            try:
                is_correct = hybrid_math_grading(answer, gt_str)
                evaluation_response = "Traditional hybrid math grading"
            except Exception as e:
                is_correct = False
                evaluation_response = f"Traditional grading error: {str(e)}"
        evaluation_method = "traditional"
    
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
        print(f"========================================================")
        
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
            "data_source": "numina_math"  # Add data source identifier
        }
        
        # Add OTC information if available
        if otc_info:
            result["otc_info"] = otc_info
            
        return result
        
    return final_score


if __name__ == '__main__':
    print("=== Testing Enhanced Numina Math Reward Function ===")
    
    # Test case 1: Correct answer with xverify
    solution1 = r"The answer is $\frac{1}{2}$. So we have \boxed{\frac{1}{2}}"
    gt1 = "1/2"
    question1 = "What is one half expressed as a fraction?"
    score1 = compute_score(solution1, gt1, question=question1, return_dict=True, use_xverify=True)
    print(f"Test 1 (xverify correct): {score1}")

    # Test case 2: Traditional grading fallback
    solution2 = "<answer>0.5</answer>"
    gt2 = "1/2"
    score2 = compute_score(solution2, gt2, return_dict=True, use_xverify=False)
    print(f"Test 2 (traditional grading): {score2}")

    # Test case 3: With OTC-PPO (no correct_trajectories needed)
    solution3 = r"""<search>mathematical computation</search>
    Let me calculate: \boxed{2}"""
    gt3 = "2"
    question3 = "What is 1+1?"
    score3 = compute_score(solution3, gt3, question=question3, return_dict=True, use_otc=True, otc_method="ppo")
    print(f"Test 3 (with OTC-PPO): {score3}")
    
    # Test case 4: With OTC-GRPO (requires correct_trajectories)
    solution4 = r"""<search>mathematical computation</search>
    <code>print("calculating")</code>
    Let me calculate: \boxed{2}"""
    gt4 = "2"
    question4 = "What is 1+1?"
    correct_trajectories = ["<search>1+1</search>\nThe answer is 2", "2"]  # Optimal uses 1 tool
    score4 = compute_score(solution4, gt4, question=question4, return_dict=True, use_otc=True, 
                          otc_method="grpo", correct_trajectories=correct_trajectories)
    print(f"Test 4 (with OTC-GRPO): {score4}")
    
    # Test case 5: Wrong answer with OTC-PPO (should be 0)
    solution5 = r"""<search>mathematical computation</search>
    Let me calculate: \boxed{3}"""  # Wrong answer
    gt5 = "2"
    question5 = "What is 1+1?"
    score5 = compute_score(solution5, gt5, question=question5, return_dict=True, use_otc=True, otc_method="ppo")
    print(f"Test 5 (wrong answer with OTC-PPO): {score5}")
    
    print("=== Enhanced Numina Math Tests Complete ===")
