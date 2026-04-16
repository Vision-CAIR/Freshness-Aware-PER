"""
测试 NQ Search 环境的 metrics 结构

验证 info 字典中的 'metrics' 键在所有情况下都正确返回
这是为了确保与 ROLL 框架的 env_manager 兼容
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig


def test_metrics_structure():
    """验证所有情况下的 info['metrics'] 结构"""
    print("=" * 80)
    print("测试 NQ Search 环境 metrics 结构")
    print("=" * 80)
    
    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=5,
        disable_limiter=True
    )
    
    env = NQSearchEnv(config)
    obs, reset_info = env.reset(seed=42)
    print("✓ 环境重置成功\n")
    print(f"当前问题: {env.current_question_data['question']}")
    print(f"金标答案: {env.current_question_data['golden_answers']}\n")
    
    # 测试用例：涵盖所有可能的代码路径
    test_cases = [
        {
            "name": "无效动作",
            "action": "this is an invalid action without proper tags",
            "expected_valid": False,
            "expected_effective": False,
            "should_terminate": False
        },
        {
            "name": "纯思考动作",
            "action": "<think>Let me think about this question</think>",
            "expected_valid": True,
            "expected_effective": True,
            "should_terminate": False
        },
        {
            "name": "思考+搜索动作",
            "action": "<think>I need to search for information</think>\n<search>test query</search>",
            "expected_valid": True,
            "expected_effective": True,
            "should_terminate": False
        },
        {
            "name": "提供答案动作",
            "action": "<think>I believe the answer is</think>\n<answer>test answer</answer>",
            "expected_valid": True,
            "expected_effective": True,
            "should_terminate": True
        },
    ]
    
    all_passed = True
    test_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"测试用例 {i}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"动作: {test_case['action'][:80]}...")
        
        obs, reward, terminated, truncated, info = env.step(test_case['action'])
        
        # 1. 验证 'metrics' 键存在
        if 'metrics' not in info:
            print(f"  ✗ 失败: info 中缺少 'metrics' 键")
            print(f"  Info keys: {list(info.keys())}")
            all_passed = False
            test_results.append({"test": test_case['name'], "passed": False, "error": "Missing 'metrics' key"})
            continue
        
        metrics = info['metrics']
        print(f"  ✓ info['metrics'] 存在")
        
        # 2. 验证 metrics 的必需字段
        required_fields = ['action_is_valid', 'action_is_effective', 'success']
        missing_fields = [field for field in required_fields if field not in metrics]
        
        if missing_fields:
            print(f"  ✗ 失败: metrics 缺少字段: {missing_fields}")
            print(f"  Metrics keys: {list(metrics.keys())}")
            all_passed = False
            test_results.append({"test": test_case['name'], "passed": False, "error": f"Missing fields: {missing_fields}"})
            continue
        
        print(f"  ✓ metrics 包含所有必需字段")
        
        # 3. 验证字段值的类型
        for field in required_fields:
            if not isinstance(metrics[field], bool):
                print(f"  ✗ 失败: metrics['{field}'] 不是布尔类型: {type(metrics[field])}")
                all_passed = False
                test_results.append({"test": test_case['name'], "passed": False, "error": f"{field} is not bool"})
                continue
        
        print(f"  ✓ 所有字段类型正确")
        
        # 4. 验证字段值的正确性
        print(f"\n  Metrics 内容:")
        print(f"    - action_is_valid: {metrics['action_is_valid']} (期望: {test_case['expected_valid']})")
        print(f"    - action_is_effective: {metrics['action_is_effective']} (期望: {test_case['expected_effective']})")
        print(f"    - success: {metrics['success']}")
        
        if metrics['action_is_valid'] != test_case['expected_valid']:
            print(f"  ⚠ 警告: action_is_valid 值不符合预期")
        
        if metrics['action_is_effective'] != test_case['expected_effective']:
            print(f"  ⚠ 警告: action_is_effective 值不符合预期")
        
        # 5. 验证终止状态
        print(f"\n  终止状态:")
        print(f"    - terminated: {terminated} (期望: {test_case['should_terminate']})")
        print(f"    - truncated: {truncated}")
        print(f"    - reward: {reward}")
        
        if terminated != test_case['should_terminate']:
            print(f"  ⚠ 警告: terminated 状态不符合预期")
        
        print(f"\n  ✓ 测试用例通过")
        test_results.append({"test": test_case['name'], "passed": True})
        
        # 如果终止了，需要重置环境
        if terminated:
            obs, reset_info = env.reset(seed=42)
            print(f"  → 环境已重置\n")
    
    # 额外测试：超过最大搜索次数
    print(f"{'='*80}")
    print(f"测试用例 {len(test_cases)+1}: 超过最大搜索次数")
    print(f"{'='*80}")
    
    obs, reset_info = env.reset(seed=42)
    for i in range(config.max_search_calls + 1):
        action = f"<think>Search {i+1}</think>\n<search>query {i+1}</search>"
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'metrics' not in info:
            print(f"  ✗ 失败: 搜索 {i+1} 时缺少 metrics")
            all_passed = False
            break
        
        print(f"  搜索 {i+1}: terminated={terminated}, metrics存在={('metrics' in info)}")
        
        if terminated:
            print(f"  ✓ 达到最大搜索次数后正确终止")
            break
    
    # 额外测试：超过最大步数
    print(f"\n{'='*80}")
    print(f"测试用例 {len(test_cases)+2}: 超过最大步数")
    print(f"{'='*80}")
    
    obs, reset_info = env.reset(seed=42)
    for i in range(config.max_steps + 1):
        action = f"<think>Step {i+1}</think>"
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'metrics' not in info:
            print(f"  ✗ 失败: 步骤 {i+1} 时缺少 metrics")
            all_passed = False
            break
        
        print(f"  步骤 {i+1}: truncated={truncated}, metrics存在={('metrics' in info)}")
        
        if truncated:
            print(f"  ✓ 达到最大步数后正确截断")
            break
    
    # 最终总结
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")
    
    for result in test_results:
        status = "✓ 通过" if result['passed'] else f"✗ 失败: {result.get('error', 'Unknown')}"
        print(f"{result['test']}: {status}")
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓✓✓ 所有测试通过！环境 metrics 结构正确，可以用于训练")
        print("=" * 80)
        return 0
    else:
        print("✗✗✗ 部分测试失败，请检查环境实现")
        print("=" * 80)
        return 1


def test_compatibility_with_env_manager():
    """测试与 env_manager 的兼容性
    
    env_manager 会访问 info['metrics'].get("action_is_valid", True)
    """
    print("\n" + "=" * 80)
    print("测试与 env_manager 的兼容性")
    print("=" * 80)
    
    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
        max_instances=2,
        disable_limiter=True
    )
    
    env = NQSearchEnv(config)
    obs, _ = env.reset(seed=42)
    
    # 模拟 env_manager 的访问模式
    action = "<think>test</think>"
    obs, reward, terminated, truncated, info = env.step(action)
    
    try:
        # 这是 env_manager 中的访问方式
        action_is_valid = info['metrics'].get("action_is_valid", True)
        print(f"✓ env_manager 访问模式测试通过")
        print(f"  info['metrics'].get('action_is_valid', True) = {action_is_valid}")
        return True
    except Exception as e:
        print(f"✗ env_manager 访问模式测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行主要测试
    exit_code = test_metrics_structure()
    
    # 运行兼容性测试
    if test_compatibility_with_env_manager():
        print("\n✓ 与 ROLL 框架 env_manager 完全兼容")
    else:
        print("\n✗ 与 ROLL 框架 env_manager 不兼容")
        exit_code = 1
    
    sys.exit(exit_code)

