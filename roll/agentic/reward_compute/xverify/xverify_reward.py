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
from typing import Dict, Any, Optional, Union
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt


# Default xverify configuration
DEFAULT_XVERIFY_CONFIG = {
    'model_name': 'xverify',
    'api_base': ['http://172.26.104.240:30033/v1'],
    'temperature': 0,
    'max_tokens': 4096,
    'api_key': 'EMPTY'
}


def extract_final_answer(solution_str: str) -> str:
    """
    Extract the final answer from model output by removing reasoning process.
    Prioritizes content after </think> tag, then falls back to the full response.
    
    Args:
        solution_str: The complete model response
        
    Returns:
        The final answer part without reasoning
    """
    # Remove content before </think> tag if present
    if '</think>' in solution_str:
        final_response = solution_str.split('</think>')[-1].strip()
    else:
        final_response = solution_str.strip()
    
    return final_response


def format_xverify_prompt(query: str, generated_output: str, ground_truth: str) -> str:
    """
    Construct the evaluation prompt for xverify model.
    
    Args:
        query: The original question
        generated_output: The model's generated answer
        ground_truth: The correct answer
        
    Returns:
        Formatted prompt for xverify evaluation
    """
    prompt = f'''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{query}"""

Output sentence: """{generated_output}"""

Correct answer: {ground_truth}

Judgement:
'''
    return prompt


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def call_xverify_model(prompt: str, model_config: Dict[str, Any]) -> str:
    """
    Call xverify model with retry mechanism.
    
    Args:
        prompt: The evaluation prompt
        model_config: Model configuration including API endpoints
        
    Returns:
        Model response string
        
    Raises:
        Exception: If all retry attempts fail
    """
    model_url = random.choice(model_config['api_base'])
    llm = OpenAI(base_url=f"{model_url}", api_key=model_config['api_key'])
    
    completion = llm.chat.completions.create(
        model=model_config['model_name'],
        messages=[{"role": "user", "content": prompt}],
        stop=['<|eot_id|>'],
        temperature=model_config['temperature'],
        max_tokens=model_config['max_tokens']
    )
    
    return completion.choices[0].message.content


def parse_xverify_response(response: str) -> tuple[Optional[int], str]:
    """
    Parse xverify model response and convert to score.
    
    Args:
        response: Raw response from xverify model
        
    Returns:
        Tuple of (score, cleaned_response):
        - score: 2 for correct, 0 for incorrect, None for invalid
        - cleaned_response: Cleaned response string
    """
    cleaned_response = response.strip().lower()
    
    # Handle different response formats
    if "correct" in cleaned_response and "incorrect" not in cleaned_response:
        return 2, response.strip()
    elif "incorrect" in cleaned_response:
        return 0, response.strip()
    else:
        # Invalid response format
        return None, response.strip()


def xverify_evaluate_single(question: str, 
                          model_answer: str, 
                          ground_truth: str,
                          model_config: Optional[Dict[str, Any]] = None) -> tuple[Optional[int], str]:
    """
    Evaluate a single sample using xverify model.
    
    Args:
        question: The original question
        model_answer: Model's generated answer
        ground_truth: Ground truth answer
        model_config: xverify model configuration
        
    Returns:
        Tuple of (score, evaluation_response):
        - score: 2 for correct, 0 for incorrect, None for error
        - evaluation_response: Raw xverify response or error message
    """
    if model_config is None:
        model_config = DEFAULT_XVERIFY_CONFIG
    
    try:
        # Extract final answer (remove reasoning process)
        final_response = extract_final_answer(model_answer)
        
        # Construct evaluation prompt
        prompt = format_xverify_prompt(question, final_response, ground_truth)
        
        # Call xverify model
        response = call_xverify_model(prompt, model_config)
        
        # Parse response and get score
        score, cleaned_response = parse_xverify_response(response)
        
        return score, cleaned_response
        
    except Exception as e:
        error_msg = f"Error in xverify evaluation: {str(e)}"
        return None, error_msg


def compute_score(solution_str: str, 
                 ground_truth: str,
                 question: str = "",
                 model_config: Optional[Dict[str, Any]] = None,
                 format_score: float = 0.0,
                 correct_score: float = 2.0,
                 return_dict: bool = False,
                 extra_info: Optional[Dict[str, Any]] = None) -> Union[float, Dict[str, Any]]:
    """
    Compute xverify-based reward score for a solution.
    
    This function implements the xverify evaluation system for AI-Researcher project,
    supporting 17 mainstream benchmarks including AIME, AMC, Browse-Comp, MATH-500,
    GPQA, GAIA, HOTPOT-QA, MINERVA, MusiQue, NQ, Sync, ToolHop, WebWalkerQA, etc.
    
    Args:
        solution_str: The model's generated solution string
        ground_truth: The ground truth answer
        question: The original question (required for xverify)
        model_config: xverify model configuration
        format_score: Score for invalid/error cases (default: 0.0)
        correct_score: Score for correct answers (default: 2.0)
        return_dict: Whether to return detailed results as dict
        extra_info: Additional information (may contain question if not provided)
        
    Returns:
        float or dict: The computed score or detailed results
    """
    # Handle case where question is passed via extra_info
    if not question and extra_info and 'question' in extra_info:
        question = extra_info['question']
    
    # 10% random sampling for debugging
    do_print = random.random() < 0.1
    
    if do_print:
        print(f"======== XVERIFY REWARD DEBUG (10% sample) ========")
        print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
        print(f"Ground truth: {ground_truth}")
        print(f"Solution preview: {solution_str[:300]}...")
        if len(solution_str) > 300:
            print(f"Solution ending: ...{solution_str[-100:]}")
    
    # Validate inputs
    if not question:
        if do_print:
            print("Warning: No question provided for xverify evaluation")
        result_score = format_score
        is_correct = False
        evaluation_response = "Error: No question provided"
    else:
        # Perform xverify evaluation
        score, evaluation_response = xverify_evaluate_single(
            question, solution_str, ground_truth, model_config
        )
        
        if score is None:
            # Error case
            result_score = format_score
            is_correct = False
        elif score == 2:
            # Correct case
            result_score = correct_score
            is_correct = True
        else:
            # Incorrect case (score == 0)
            result_score = format_score
            is_correct = False
    
    if do_print:
        print(f"Xverify evaluation: {evaluation_response}")
        print(f"Final score: {result_score} (Correct: {is_correct})")
        print(f"=================================================")
    
    if return_dict:
        return {
            "score": result_score,
            "is_correct": is_correct,
            "evaluation_response": evaluation_response,
            "extracted_answer": extract_final_answer(solution_str),
            "model_config": model_config or DEFAULT_XVERIFY_CONFIG,
            "has_search_pattern": False,  # Consistency with other reward functions
            "has_execution_pattern": False  # Consistency with other reward functions
        }
    
    return result_score


def batch_evaluate(data_list: list, 
                  model_config: Optional[Dict[str, Any]] = None) -> list:
    """
    Batch evaluate multiple samples using xverify.
    
    Args:
        data_list: List of dicts containing 'query', 'response', 'gt' fields
        model_config: xverify model configuration
        
    Returns:
        List of results with added 'score' and 'evaluation' fields
    """
    results = []
    for item in data_list:
        score, evaluation = xverify_evaluate_single(
            item['query'], 
            item['response'], 
            item['gt'], 
            model_config
        )
        
        item['score'] = score
        item['evaluation'] = evaluation
        results.append(item)
    
    return results


if __name__ == '__main__':
    print("=== Testing Xverify Reward Function ===")
    
    # Test case 1: Simple correct answer
    solution1 = "<think>Let me calculate: 2+2=4</think>The answer is 4."
    gt1 = "4"
    question1 = "What is 2+2?"
    score1 = compute_score(solution1, gt1, question1, return_dict=True)
    print(f"Test 1 (Simple correct): {score1}")
    
    # Test case 2: Mathematical equivalence
    solution2 = "<think>Converting to polar coordinates...</think>The answer is (3, π/2)."
    gt2 = "\\left( 3, \\frac{\\pi}{2} \\right)"
    question2 = "Convert the point (0,3) to polar coordinates."
    score2 = compute_score(solution2, gt2, question2, return_dict=True)
    print(f"Test 2 (Math equivalence): {score2}")
    
    # Test case 3: No question provided (error case)
    solution3 = "The answer is 42."
    gt3 = "42"
    question3 = ""
    score3 = compute_score(solution3, gt3, question3, return_dict=True)
    print(f"Test 3 (No question): {score3}")
    
    print("=== Xverify Tests Complete ===")
