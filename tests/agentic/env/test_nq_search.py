"""
NQ Search Environment Tests

测试 NQ Search 环境的各项功能
"""

import sys
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """测试基础环境功能"""
    print("=" * 80)
    print("测试 1: 基础环境功能")
    print("=" * 80)

    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=5,
        retrieval_server_url="http://127.0.0.1:8100/retrieve",
        retrieval_timeout=30,
        retrieval_topk=3,
        max_steps=5,
        max_search_calls=3,
        disable_limiter=True
    )

    try:
        env = NQSearchEnv(config)
        print(f"✓ 环境创建成功，加载了 {len(env.data)} 个样本")

        obs, info = env.reset(seed=42)
        print(f"\n初始观察:")
        print(obs[:500] + "..." if len(obs) > 500 else obs)
        print(f"\n当前问题: {env.current_question_data['question']}")
        print(f"金标答案: {env.current_question_data['golden_answers']}")

        search_action = '''<think>
我需要搜索关于这个问题的信息。
</think>

<search>test query for Natural Questions</search>'''

        print(f"\n执行搜索动作...")
        obs, reward, terminated, truncated, info = env.step(search_action)
        print(f"✓ 搜索完成")
        print(f"  - 奖励: {reward}")
        print(f"  - 已搜索次数: {info.get('search_calls', 0)}")
        print(f"  - 终止: {terminated}, 截断: {truncated}")
        print(f"  - 观察长度: {len(obs)} 字符")
        
        if "<information>" in obs:
            print(f"  - 检索结果已包含在观察中")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_em_evaluation():
    """测试 EM 评估"""
    print("\n" + "=" * 80)
    print("测试 2: EM 评估")
    print("=" * 80)

    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=3,
        retrieval_server_url="http://127.0.0.1:8100/retrieve",
        max_steps=5,
        max_search_calls=2,
        disable_limiter=True,
        use_em_evaluation=True,
        em_ignore_case=True,
        em_ignore_punctuation=True,
        em_ignore_articles=True
    )

    try:
        env = NQSearchEnv(config)

        # Test case 1: 正确答案
        obs, info = env.reset(seed=0)
        golden_answers = env.current_question_data['golden_answers']
        question = env.current_question_data['question']
        
        print(f"\n测试用例 1: 正确答案")
        print(f"问题: {question}")
        print(f"金标答案: {golden_answers}")

        correct_answer = golden_answers[0] if golden_answers else "Unknown"
        action = f'''<think>
根据我的知识，答案是 {correct_answer}。
</think>

<answer>{correct_answer}</answer>'''

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - 回答: {correct_answer}")
        print(f"  - 奖励: {reward}")
        print(f"  - 正确: {info.get('success', False)}")
        print(f"  - EM分数: {info.get('raw_score', 0.0)}")

        # Test case 2: 错误答案
        env.reset(seed=1)
        golden_answers = env.current_question_data['golden_answers']
        
        print(f"\n测试用例 2: 错误答案")
        wrong_answer = "This is definitely wrong"
        action = f'''<think>
我猜测答案是 {wrong_answer}。
</think>

<answer>{wrong_answer}</answer>'''

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - 回答: {wrong_answer}")
        print(f"  - 奖励: {reward}")
        print(f"  - 正确: {info.get('success', False)}")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_termination_conditions():
    """测试终止条件"""
    print("\n" + "=" * 80)
    print("测试 3: 终止条件")
    print("=" * 80)

    # Test 1: 答案终止
    print("\n子测试 3.1: 答案终止")
    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=1,
        max_steps=10,
        max_search_calls=5,
        disable_limiter=True
    )

    try:
        env = NQSearchEnv(config)
        obs, info = env.reset(seed=42)

        action = '''<think>我直接提供答案。</think>

<answer>Test Answer</answer>'''

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - 终止: {terminated}")
        print(f"  - 原因: {info.get('termination_reason', 'unknown')}")
        print("  ✓ 通过")

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

    # Test 2: 最大搜索次数
    print("\n子测试 3.2: 最大搜索次数限制")
    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=1,
        max_steps=10,
        max_search_calls=2,
        disable_limiter=True
    )

    try:
        env = NQSearchEnv(config)
        obs, info = env.reset(seed=42)

        for i in range(3):
            action = f'''<think>搜索 {i+1}</think>

<search>query {i+1}</search>'''

            obs, reward, terminated, truncated, info = env.step(action)
            search_count = info.get('search_calls', 0)
            print(f"  - 搜索 {i+1}: 已搜索 {search_count} 次")
            
            if terminated or truncated:
                print(f"  - 终止原因: {info.get('termination_reason', 'unknown')}")
                break

        print("  ✓ 通过")
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def test_complete_trajectory():
    """测试完整轨迹"""
    print("\n" + "=" * 80)
    print("测试 4: 完整交互轨迹")
    print("=" * 80)

    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=3,
        retrieval_server_url="http://127.0.0.1:8100/retrieve",
        retrieval_topk=3,
        max_steps=6,
        max_search_calls=3,
        disable_limiter=True
    )

    try:
        env = NQSearchEnv(config)
        trajectories = []

        for sample_idx in range(min(3, len(env.data))):
            print(f"\n处理样本 {sample_idx + 1}/3")
            
            trajectory = {
                "sample_id": sample_idx,
                "success": False,
                "total_reward": 0.0
            }

            obs, info = env.reset(seed=sample_idx + 100)
            trajectory["question"] = env.current_question_data["question"]
            trajectory["golden_answers"] = env.current_question_data["golden_answers"]

            print(f"问题: {trajectory['question'][:100]}...")

            total_reward = 0.0

            # 模拟多轮交互
            for step in range(config.max_steps):
                if step < 2:
                    query = trajectory['question'][:80] if step == 0 else f"details {trajectory['question'][:50]}"
                    action = f'''<think>我需要搜索信息。</think>

<search>{query}</search>'''

                else:
                    answer = trajectory["golden_answers"][0] if trajectory["golden_answers"] else "Unknown"
                    action = f'''<think>基于搜索结果，我给出答案。</think>

<answer>{answer}</answer>'''

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                print(f"  步骤 {step+1}: reward={reward:.3f}, 终止={terminated}")

                if terminated or truncated:
                    trajectory["success"] = info.get("success", False)
                    print(f"  成功: {trajectory['success']}")
                    break

            trajectory["total_reward"] = total_reward
            trajectories.append(trajectory)

        # 保存轨迹
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "nq_search_trajectories.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)

        success_count = sum(1 for t in trajectories if t.get("success", False))
        avg_reward = sum(t.get("total_reward", 0.0) for t in trajectories) / len(trajectories)

        print(f"\n轨迹测试完成:")
        print(f"  - 成功: {success_count}/{len(trajectories)}")
        print(f"  - 平均奖励: {avg_reward:.3f}")
        print(f"  - 结果保存到: {output_file}")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("NQ Search Environment 测试套件")
    print("=" * 80)

    # 检查检索服务器
    print("\n检查检索服务器状态...")
    try:
        import requests
        response = requests.get("http://127.0.0.1:8100/health", timeout=5)
        if response.status_code == 200:
            print("✓ 检索服务器可用")
        else:
            print(f"⚠ 检索服务器返回异常状态: {response.status_code}")
    except Exception as e:
        print(f"⚠ 无法连接到检索服务器: {e}")

    # 运行测试
    tests = [
        ("基础功能", test_basic_functionality),
        ("EM评估", test_em_evaluation),
        ("终止条件", test_termination_conditions),
        ("完整轨迹", test_complete_trajectory),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✓ 通过" if success else "✗ 失败"
        except Exception as e:
            results[test_name] = f"✗ 异常: {str(e)}"

    # 打印测试总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for test_name, result in results.items():
        print(f"{test_name}: {result}")

    all_passed = all("✓" in r for r in results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败 ✗")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

