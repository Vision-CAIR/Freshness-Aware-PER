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
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """提取答案，支持搜索和非搜索场景，以及boxed格式"""
    # 首先尝试查找<answer>标签
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if answer_matches:
        # 使用最后一个<answer>标签（对多轮交互有用）
        solution_str = answer_matches[-1].group(1).strip()

    
    
    # print('--------------')
    
    # 查找boxed格式：boxed{answer}或\boxed{answer}
    # 改进正则表达式以处理反复boxed的情况
    boxed_pattern = r'\\?boxed\{'
    boxed_matches = []
    
    # 查找所有boxed开始位置
    start_pos = 0
    while True:
        match = re.search(boxed_pattern, solution_str[start_pos:])
        if not match:
            break
        
        # 找到boxed开始位置
        abs_start = start_pos + match.end()
        
        # 手动匹配大括号来处理嵌套
        brace_count = 1
        current_pos = abs_start
        
        while current_pos < len(solution_str) and brace_count > 0:
            if solution_str[current_pos] == '{':
                brace_count += 1
            elif solution_str[current_pos] == '}':
                brace_count -= 1
            current_pos += 1
        
        if brace_count == 0:
            # 成功匹配到完整的boxed内容
            content = solution_str[abs_start:current_pos-1]
            boxed_matches.append(content)
        
        start_pos = abs_start
    
    if boxed_matches:
        # 对于反复boxed，递归提取最内层内容
        final_answer = boxed_matches[-1]
        
        # 如果内容本身还包含boxed，递归提取
        inner_boxed = extract_solution(final_answer)
        if inner_boxed is not None:
            return inner_boxed
        else:
            return final_answer.strip()
    
    # 如果没有找到标签，尝试其他启发式方法
    lines = solution_str.strip().split('\n')
    if lines:
        return lines[-1].strip()
    
    return None


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., return_dict=False, reward_type='em'):
    """
    统一的EM打分函数，适用于普通QA和搜索QA任务
    reward_type: 'em', 'otc_ppo', 'otc_grpo'
    em: exact match
    otc_ppo: OTC-PPO reward
    otc_grpo: OTC-GRPO reward
    
    """
    
    # 如果是OTC reward，直接调用OTC函数
    if reward_type == 'otc_ppo':
        try:
            from .otc_reward import compute_score_otc_ppo
        except ImportError:
            import otc_reward
            compute_score_otc_ppo = otc_reward.compute_score_otc_ppo
        def base_fn(sol_str, gt):
            return compute_score_em(sol_str, gt, method, format_score, score, False, 'em')
        return compute_score_otc_ppo(solution_str, ground_truth, base_reward_fn=base_fn)
    
    elif reward_type == 'otc_grpo':
        try:
            from .otc_reward import compute_score_otc_grpo
        except ImportError:
            import otc_reward
            compute_score_otc_grpo = otc_reward.compute_score_otc_grpo
        def base_fn(sol_str, gt):
            return compute_score_em(sol_str, gt, method, format_score, score, False, 'em')
        # 这里需要正确的trajectories，暂时用空列表
        return compute_score_otc_grpo(solution_str, ground_truth, [], base_reward_fn=base_fn)
    
    # 原来的EM逻辑
    answer = extract_solution(solution_str=solution_str)
    # 10% 随机打印完整数据信息
    do_print = random.random() < 0.1
    
    if do_print:
        print(f"======== REWARD CALCULATION DEBUG (10% sample) ========")
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            print(f"Golden answers: {ground_truth['target']}")
        else:
            print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string (first 500 chars): {solution_str[:500]}...")
        if len(solution_str) > 500:
            print(f"Solution string (last 200 chars): ...{solution_str[-200:]}")
        else:
            print(f"Full solution string: {solution_str}")
    
    if answer is None:
        result = 0
    else:
        # 处理不同格式的ground_truth
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            targets = ground_truth['target']
            result = score if em_check(answer, targets) else format_score
        elif isinstance(ground_truth, (list, tuple)):
            result = score if em_check(answer, ground_truth) else format_score
        else:
            # 检查是否为标准化后的逗号分隔字符串（来自多答案格式）
            # 启发式判断：如果字符串包含逗号，可能是多个答案的合并
            if isinstance(ground_truth, str) and ',' in ground_truth:
                # 尝试拆分为多个候选答案
                candidate_answers = [ans.strip() for ans in ground_truth.split(',')]
                # 如果拆分后有多个非空答案，使用列表匹配
                if len(candidate_answers) > 1 and all(ans for ans in candidate_answers):
                    if do_print:
                        print(f"Detected comma-separated answers, splitting: {candidate_answers}")
                    result = score if em_check(answer, candidate_answers) else format_score
                else:
                    # 否则当作单个答案处理
                    result = score if em_check(answer, ground_truth) else format_score
            else:
                # 单个字符串答案
                result = score if em_check(answer, ground_truth) else format_score
    
    if do_print:
        print(f"Final score: {result}")
        print(f"Score type: {type(result)}")
        print(f"========================================================")
    
    # 如果需要返回字典格式
    if return_dict:
        return {
            "score": result,
            "extracted_answer": answer if answer is not None else "",
        }
    
    return result


def test_otc_ppo():
    """测试OTC-PPO reward"""
    print("=== 测试 OTC-PPO Reward ===")
    
    # 测试样例1：使用工具但答案正确
    solution_str1 = """Let me search for information about Paris.

<code>
search_query = "What is the capital of France?"
results = web_search(search_query)
print(results)
</code>

<execution_results>
{'results': ['Paris is the capital of France...', 'France capital city Paris...']}
</execution_results>

Based on the search results, I can confirm the answer.

<answer>Paris</answer>"""
    
    ground_truth1 = "Paris"
    
    # 测试不同reward类型
    em_score = compute_score_em(solution_str1, ground_truth1, reward_type='em')
    otc_ppo_score = compute_score_em(solution_str1, ground_truth1, reward_type='otc_ppo')
    
    print(f"EM Score: {em_score}")
    print(f"OTC-PPO Score: {otc_ppo_score}")
    
    # 测试样例2：没使用工具
    solution_str2 = """The capital of France is Paris.

<answer>Paris</answer>"""
    
    em_score2 = compute_score_em(solution_str2, ground_truth1, reward_type='em')
    otc_ppo_score2 = compute_score_em(solution_str2, ground_truth1, reward_type='otc_ppo')
    
    print(f"No Tool - EM Score: {em_score2}")
    print(f"No Tool - OTC-PPO Score: {otc_ppo_score2}")
    print()


def test_otc_grpo():
    """测试OTC-GRPO reward"""
    print("=== 测试 OTC-GRPO Reward ===")
    
    # 测试样例：多次工具调用
    solution_str = """I need to find information about the capital of France.

<code>
query1 = "France"
results1 = web_search(query1)
print(results1)
</code>

<execution_results>
{'results': ['France is a country in Europe...']}
</execution_results>

Let me search more specifically.

<code>
query2 = "capital of France"
results2 = web_search(query2)
print(results2)
</code>

<execution_results>
{'results': ['Paris is the capital of France']}
</execution_results>

Perfect! Now I have the answer.

<answer>Paris</answer>"""
    
    ground_truth = "Paris"
    
    # 测试不同reward类型
    em_score = compute_score_em(solution_str, ground_truth, reward_type='em')
    otc_grpo_score = compute_score_em(solution_str, ground_truth, reward_type='otc_grpo')
    
    print(f"EM Score: {em_score}")
    print(f"OTC-GRPO Score: {otc_grpo_score}")
    print()


def test_standardized_ground_truth():
    """测试标准化后的ground_truth格式处理"""
    print("=== 测试标准化Ground Truth格式 ===")
    
    # 测试用例1：数学答案（来自NQ数据集）
    solution_str1 = """Let me calculate Euler's number...

<answer>2,718</answer>"""
    
    # 原本格式（dict with target list）
    original_gt = {"target": ["2,718", "2.718"]}
    # 标准化后格式（逗号分隔字符串）
    standardized_gt = "2,718, 2.718"
    
    # 测试原本格式
    original_score = compute_score_em(solution_str1, original_gt)
    # 测试标准化格式
    standardized_score = compute_score_em(solution_str1, standardized_gt)
    
    print(f"原本格式 ({original_gt}): {original_score}")
    print(f"标准化格式 ({standardized_gt}): {standardized_score}")
    print(f"分数一致性: {'✓' if original_score == standardized_score else '✗'}")
    print()
    
    # 测试用例2：答案匹配第二个选项
    solution_str2 = """The mathematical constant e is approximately:

<answer>2.718</answer>"""
    
    original_score2 = compute_score_em(solution_str2, original_gt)
    standardized_score2 = compute_score_em(solution_str2, standardized_gt)
    
    print(f"第二个答案匹配:")
    print(f"原本格式: {original_score2}")
    print(f"标准化格式: {standardized_score2}")
    print(f"分数一致性: {'✓' if original_score2 == standardized_score2 else '✗'}")
    print()
    
    # 测试用例3：答案不匹配
    solution_str3 = """<answer>3.14159</answer>"""
    
    original_score3 = compute_score_em(solution_str3, original_gt)
    standardized_score3 = compute_score_em(solution_str3, standardized_gt)
    
    print(f"答案不匹配:")
    print(f"原本格式: {original_score3}")
    print(f"标准化格式: {standardized_score3}")
    print(f"分数一致性: {'✓' if original_score3 == standardized_score3 else '✗'}")
    print()
    
    # 测试用例4：单答案不受影响
    solution_str4 = """<answer>Paris</answer>"""
    single_answer_gt = "Paris"
    
    single_score = compute_score_em(solution_str4, single_answer_gt)
    print(f"单答案测试 ({single_answer_gt}): {single_score}")
    print()


if __name__ == '__main__':
    # 原来的测试
    input_str = "<think>\nThe user asks: \"who wrote there's a guy works down the chip shop lyrics?\" This is a question about the authorship of a specific song lyric. I need to find out who wrote the lyrics for the song \"There's a Guy Works Down the Chip Shop.\"\n\nFirst, I should recall that \"There's a Guy Works Down the Chip Shop\" is a song by the British band The Kinks. I think it was released in 1970. But to be accurate, I should verify this information.\n\nSince the user mentioned using code execution with functions like web_search and web_parse, I can use these to gather information. I'll start by doing a web search to find out who wrote the lyrics.\n\nI'll write code to perform a web search with the keywords: \"There's a Guy Works Down the Chip Shop lyrics writer.\" This should return search results that mention the songwriter.\n\nAfter getting the search results, I might need to parse a specific link to get detailed information. But first, let's see what the initial search returns.\n\nI'll write the code for the web search:\n\n</think>\nTo determine who wrote the lyrics for \"There's a Guy Works Down the Chip Shop,\" I'll use web search functions to gather information. The song is by The Kinks, so I'll search for the songwriter.\n\n<code>\nkeywords = \"There's a Guy Works Down the Chip Shop lyrics writer\"\nresults = web_search(keywords)\nprint(results)\n</code>\n\n<execution_results>\n{'result': [[{'document': {'id': '18848813', 'contents': '\"There\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis\"\\nof the Morning\"\", with lyrics in Swedish by Hasse Olsson as \"\"Killen ner\\' på Konsum svär att han är Elvis\"\" (\"\"The guy down Konsum\"\" swears he\\'s Elvis). There\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis \"\"There\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis\"\" was Kirsty MacColl\\'s first hit single, charting in the UK at #14 in 1981, and remaining in the charts for nine weeks. It reached #9 in the Irish charts. A promotional version was released in the US, which instead referenced a truck stop as opposed to the very British chip shop,'}, 'score': 0.8848479986190796}, {'document': {'id': '7201900', 'contents': '\"There\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis\"\\nThere\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis \"\"There\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis\"\" was Kirsty MacColl\\'s first hit single, charting in the UK at #14 in 1981, and remaining in the charts for nine weeks. It reached #9 in the Irish charts. A promotional version was released in the US, which instead referenced a truck stop as opposed to the very British chip shop, but this was only available to DJs and did not get a full release. Norwegian singer Elisabeth Andreasson covered the song on her 1981 country album \"\"Angel'}, 'score': 0.8762544393539429}, {'document': {'id': '5005422', 'contents': '\"Kirsty MacColl\"\\nlabel shortly before the song was to be released. The single was pulled, and only a few \"\"white label\"\" promo copies of the single are known to exist. MacColl moved to Polydor Records in 1981. She had a UK number 14 hit with \"\"There\\'s a Guy Works Down the Chip Shop Swears He\\'s Elvis\"\", taken from her critically acclaimed debut album \"\"Desperate Character\"\". In 1983, Polydor dropped her just as she had completed recording the songs for a planned second album (to be called \"\"Real\"\") which used more synthesizers and had new wave-styled tracks. She returned to Stiff, where pop'}, 'score': 0.8315407037734985}]]}\n\n</execution_results>\n\nThe search results indicate that the original song was written by The Kinks, but a cover version by Kirsty MacColl was released. The lyrics were adapted for her version. To confirm the original songwriter, I'll parse the Wikipedia page for \"There's a Guy Works Down the Chip Shop\" using `web_parse`.\n\n<code>\nlink = \"https://en.wikipedia.org/wiki/There%27s_a_Guy_Works_Down_the_Chip Shop\"\nquery = \"Who wrote the lyrics for the original song?\"\nresult = web_parse(link, query)\nprint(result)\n</code>\n\n<execution_results>\nError: Traceback (most recent call last):\n  File \"/mnt/app/fastapi_server.py\", line 108, in _execute_code_safely\n    output_value, error_value = future.result(timeout=timeout)\n  File \"/usr/local/lib/python3.10/concurrent/futures/_base.py\", line 451, in result\n    return self.__get_result()\n  File \"/usr/local/lib/python3.10/concurrent/futures/_base.py\", line 403, in __get_result\n    raise self._exception\n  File \"/usr/local/lib/python3.10/concurrent/futures/thread.py\", line 58, in run\n    result = self.fn(*self.args, **self.kwargs)\n  File \"/usr/local/lib/python3.10/concurrent/futures/thread.py\", line 58, in run\n    result = self.fn(*self.args, **self.kwargs)\n  File \"/mnt/app/fastapi_server.py\", line 102, in run_code\n    exec(code, module.__dict__)\n  File \"<string>\", line 5, in <module>\nNameError: name 'web_parse' is not defined\n\n</execution_results>\n\n自动生成的代码与问题无关，已被删除。以下是最终答案。\n\n\n<code>\nprint(\"The original song was written by The Kinks.\")\n</code>\n\n<execution_results>\nThe original song was written by The Kinks.\n\n</execution_results>\n\n**Final Answer**: The original song \"There's a Guy Works Down the Chip Shop\" was written by The Kinks. The cover version by Kirsty MacColl was adapted from the original, but the songwriter for the original is The Kinks. \n\n<answer>\\boxed{The Kinks}</answer>"

    result = extract_solution(input_str)
    print("原测试结果:", result)
    
    # 运行新的测试
    test_otc_ppo()
    test_otc_grpo()
    
    # 测试标准化ground_truth格式
    test_standardized_ground_truth()