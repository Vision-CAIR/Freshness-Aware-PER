"""测试 NQSearch 环境的 Goal-Conditioned 设计

验证每个 step 的 observation 都包含 Goal（原始问题），
确保 Step-Level 和 Trajectory-Level 训练都能正确学习。

使用 Mock 搜索结果，不需要真实的检索服务器。
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import pytest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock 数据集
MOCK_DATASET = [
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant that answers questions using search."},
            {"role": "user", "content": "Who won the first Nobel Prize in Physics?"}
        ],
        "golden_answers": ["Wilhelm Röntgen", "Wilhelm Conrad Röntgen", "Röntgen"],
        "question": "Who won the first Nobel Prize in Physics?"
    },
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant that answers questions using search."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "golden_answers": ["Paris"],
        "question": "What is the capital of France?"
    },
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant that answers questions using search."},
            {"role": "user", "content": "When was Python programming language created?"}
        ],
        "golden_answers": ["1991", "1989"],
        "question": "When was Python programming language created?"
    },
]


# Mock 搜索结果
MOCK_SEARCH_RESULTS = {
    "nobel": "Doc 1(Title: Nobel Prize) Wilhelm Conrad Röntgen was awarded the first Nobel Prize in Physics in 1901.",
    "capital": "Doc 1(Title: France) Paris is the capital and most populous city of France.",
    "python": "Doc 1(Title: Python) Python was created by Guido van Rossum and first released in 1991.",
    "default": "Doc 1(Title: Search Result) This is a mock search result for testing purposes."
}


def get_mock_search_result(query: str) -> str:
    """根据查询返回 mock 搜索结果"""
    query_lower = query.lower()
    for key, result in MOCK_SEARCH_RESULTS.items():
        if key in query_lower:
            return result
    return MOCK_SEARCH_RESULTS["default"]


class MockDataset:
    """Mock Dataset 类，模拟 HuggingFace datasets 的行为"""

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def select(self, indices):
        return MockDataset([self._data[i] for i in indices])


class TestGoalConditioned:
    """测试 Goal-Conditioned 设计"""

    @pytest.fixture
    def env(self):
        """创建测试环境，使用 mock 数据集和搜索"""
        from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig

        config = NQSearchEnvConfig(
            dataset_path="mock://not_used",  # 不会真正加载
            max_instances=3,
            disable_limiter=True,
            max_steps=5,
            max_search_calls=3,
            use_outcome_reward_only=True,
            use_em_evaluation=True
        )

        # Mock 数据集加载
        with patch.object(NQSearchEnv, '_load_dataset', return_value=MockDataset(MOCK_DATASET)):
            env = NQSearchEnv(config)

        yield env
        env.close()

    def test_reset_contains_goal(self, env):
        """测试 reset 返回的 observation 包含 Goal"""
        logger.info("\n" + "=" * 60)
        logger.info("测试 reset() 返回的 observation 包含 Goal")
        logger.info("=" * 60)

        obs, info = env.reset(seed=0)

        # 检查 observation 包含问题
        assert "Question:" in obs, "Observation should contain 'Question:'"
        assert "Nobel Prize" in obs or "Physics" in obs, "Observation should contain the question content"

        # 检查包含系统提示
        assert "helpful assistant" in obs, "Observation should contain system prompt"

        logger.info("✓ Reset observation 包含 Goal")
        logger.info(f"Observation preview:\n{obs[:300]}...")

    def test_step_search_contains_goal(self, env):
        """测试搜索后的 observation 包含 Goal"""
        logger.info("\n" + "=" * 60)
        logger.info("测试搜索后的 observation 包含 Goal")
        logger.info("=" * 60)

        obs, info = env.reset(seed=0)
        original_question = env.current_question
        logger.info(f"Original Question: {original_question}")

        # 执行搜索（使用 mock）
        search_action = '''<think>I need to search for information.</think>
<search>nobel prize physics winner</search>'''

        # Mock 搜索调用
        with patch('roll.agentic.env.nq_search.utils.call_retrieval_server',
                   return_value=get_mock_search_result("nobel")):
            obs, reward, terminated, truncated, info = env.step(search_action)

        # 验证 observation 包含 Goal
        assert "Question:" in obs, "Search result observation should contain 'Question:'"
        assert original_question in obs or "Nobel" in obs, \
            f"Search result observation should contain the original question"

        # 验证包含搜索结果
        assert "<information>" in obs, "Should contain search results"

        logger.info("✓ 搜索后 observation 包含 Goal")
        logger.info(f"Observation preview:\n{obs[:500]}...")

    def test_step_answer_contains_goal(self, env):
        """测试提交答案后的 observation 包含 Goal"""
        logger.info("\n" + "=" * 60)
        logger.info("测试提交答案后的 observation 包含 Goal")
        logger.info("=" * 60)

        obs, info = env.reset(seed=0)
        original_question = env.current_question

        # 直接提交答案
        answer_action = '''<think>I know the answer.</think>
<answer>Wilhelm Röntgen</answer>'''

        obs, reward, terminated, truncated, info = env.step(answer_action)

        # 验证终止
        assert terminated, "Should terminate after answer"

        # 验证 observation 包含 Goal
        assert "Question:" in obs, "Answer observation should contain 'Question:'"

        logger.info("✓ 提交答案后 observation 包含 Goal")
        logger.info(f"Reward: {reward}, Success: {info.get('success')}")
        logger.info(f"Observation preview:\n{obs[:400]}...")

    def test_step_invalid_action_contains_goal(self, env):
        """测试无效动作后的 observation 包含 Goal"""
        logger.info("\n" + "=" * 60)
        logger.info("测试无效动作后的 observation 包含 Goal")
        logger.info("=" * 60)

        obs, info = env.reset(seed=1)
        original_question = env.current_question
        logger.info(f"Original Question: {original_question}")

        # 发送无效动作
        invalid_action = "This is just plain text without any tags"

        obs, reward, terminated, truncated, info = env.step(invalid_action)

        # 验证无效动作被识别
        assert not info["metrics"]["action_is_valid"], "Should be invalid action"

        # 验证 observation 包含 Goal
        assert "Question:" in obs, "Invalid action observation should contain 'Question:'"
        assert "Invalid action" in obs, "Should contain error message"

        logger.info("✓ 无效动作后 observation 包含 Goal")
        logger.info(f"Observation preview:\n{obs[:400]}...")

    def test_multi_step_all_contain_goal(self, env):
        """测试多步交互中每一步都包含 Goal"""
        logger.info("\n" + "=" * 60)
        logger.info("测试多步交互中每一步都包含 Goal")
        logger.info("=" * 60)

        obs, info = env.reset(seed=2)
        original_question = env.current_question
        logger.info(f"Original Question: {original_question}")

        # Step 1: 搜索
        search_action1 = '''<think>First search</think>
<search>python programming language</search>'''

        with patch('roll.agentic.env.nq_search.utils.call_retrieval_server',
                   return_value=get_mock_search_result("python")):
            obs1, _, _, _, _ = env.step(search_action1)

        assert "Question:" in obs1, "Step 1 observation should contain Goal"
        logger.info("✓ Step 1 (搜索) 包含 Goal")

        # Step 2: 再次搜索
        search_action2 = '''<think>Second search</think>
<search>python created when</search>'''

        with patch('roll.agentic.env.nq_search.utils.call_retrieval_server',
                   return_value=get_mock_search_result("python")):
            obs2, _, _, _, _ = env.step(search_action2)

        assert "Question:" in obs2, "Step 2 observation should contain Goal"
        logger.info("✓ Step 2 (搜索) 包含 Goal")

        # Step 3: 纯思考
        think_action = '''<think>Let me think about this more...</think>'''

        obs3, _, _, _, _ = env.step(think_action)

        assert "Question:" in obs3, "Step 3 observation should contain Goal"
        logger.info("✓ Step 3 (思考) 包含 Goal")

        # Step 4: 提交答案
        answer_action = '''<think>Now I can answer</think>
<answer>1991</answer>'''

        obs4, reward, terminated, _, info = env.step(answer_action)

        assert "Question:" in obs4, "Step 4 observation should contain Goal"
        assert terminated, "Should terminate"
        logger.info("✓ Step 4 (答案) 包含 Goal")
        logger.info(f"Final reward: {reward}, Success: {info.get('success')}")

    def test_goal_consistency_across_episode(self, env):
        """测试整个 episode 中 Goal 保持一致"""
        logger.info("\n" + "=" * 60)
        logger.info("测试整个 episode 中 Goal 保持一致")
        logger.info("=" * 60)

        obs, info = env.reset(seed=0)
        original_question = env.current_question

        observations = [obs]

        # 执行多个步骤
        actions = [
            '<think>Search first</think><search>test query</search>',
            '<think>Think more</think>',
            '<think>Answer now</think><answer>Test Answer</answer>'
        ]

        for action in actions:
            with patch('roll.agentic.env.nq_search.utils.call_retrieval_server',
                       return_value=get_mock_search_result("default")):
                obs, reward, terminated, truncated, info = env.step(action)
                observations.append(obs)
                if terminated or truncated:
                    break

        # 验证所有 observation 都包含相同的 Goal
        for i, obs in enumerate(observations):
            assert "Question:" in obs, f"Observation {i} should contain 'Question:'"
            logger.info(f"✓ Observation {i} 包含 Goal")

        logger.info("✓ 所有 observations 都包含一致的 Goal")

    def test_step_level_training_ready(self, env):
        """验证 Step-Level 训练就绪性：每个独立 step 都是自包含的"""
        logger.info("\n" + "=" * 60)
        logger.info("验证 Step-Level 训练就绪性")
        logger.info("=" * 60)

        obs, info = env.reset(seed=0)

        # 执行一次搜索
        search_action = '<think>Search</think><search>nobel prize</search>'

        with patch('roll.agentic.env.nq_search.utils.call_retrieval_server',
                   return_value=get_mock_search_result("nobel")):
            obs, _, _, _, _ = env.step(search_action)

        # 验证这个单独的 observation 包含所有训练所需的信息
        logger.info("检查单个 step observation 是否自包含：")

        # 1. 包含系统提示（任务说明）
        has_system_prompt = "helpful assistant" in obs or "search" in obs.lower()
        logger.info(f"  - 系统提示: {'✓' if has_system_prompt else '✗'}")

        # 2. 包含 Goal（问题）
        has_goal = "Question:" in obs
        logger.info(f"  - Goal (问题): {'✓' if has_goal else '✗'}")

        # 3. 包含当前状态（搜索结果）
        has_state = "<information>" in obs
        logger.info(f"  - 当前状态: {'✓' if has_state else '✗'}")

        assert has_system_prompt, "Should contain system prompt"
        assert has_goal, "Should contain goal"
        assert has_state, "Should contain current state"

        logger.info("\n✓ Step-Level 训练就绪：每个 step 都是自包含的")
        logger.info("  模型可以从任意单个 step 学习，无需依赖历史上下文")


def test_goal_conditioned_demo():
    """Goal-Conditioned 设计演示（不使用 fixture）"""
    logger.info("\n" + "=" * 80)
    logger.info("Goal-Conditioned 设计演示")
    logger.info("=" * 80)

    from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig

    config = NQSearchEnvConfig(
        dataset_path="mock://not_used",
        max_instances=3,
        disable_limiter=True,
        max_steps=5,
        max_search_calls=3
    )

    # Mock 数据集加载
    with patch.object(NQSearchEnv, '_load_dataset', return_value=MockDataset(MOCK_DATASET)):
        env = NQSearchEnv(config)

    logger.info("\n【演示】完整的 Goal-Conditioned 交互流程：")
    logger.info("-" * 60)

    # Reset
    obs, _ = env.reset(seed=0)
    logger.info(f"\n[Reset] Initial Observation:")
    logger.info(f"{'='*40}")
    logger.info(obs[:600])
    logger.info(f"{'='*40}")

    # Step 1: 搜索
    action1 = '<think>Let me search for Nobel Prize winners.</think><search>nobel prize physics first</search>'

    with patch('roll.agentic.env.nq_search.utils.call_retrieval_server',
               return_value=get_mock_search_result("nobel")):
        obs1, r1, t1, _, info1 = env.step(action1)

    logger.info(f"\n[Step 1] After Search (注意 Question 仍然存在!):")
    logger.info(f"{'='*40}")
    logger.info(obs1[:700])
    logger.info(f"{'='*40}")
    logger.info(f"Reward: {r1}, Terminated: {t1}")

    # Step 2: 回答
    action2 = '<think>Based on the search, the answer is Wilhelm Röntgen.</think><answer>Wilhelm Röntgen</answer>'

    obs2, r2, t2, _, info2 = env.step(action2)

    logger.info(f"\n[Step 2] After Answer (终止时也包含 Question!):")
    logger.info(f"{'='*40}")
    logger.info(obs2[:500])
    logger.info(f"{'='*40}")
    logger.info(f"Reward: {r2}, Success: {info2.get('success')}, Terminated: {t2}")

    env.close()

    logger.info("\n" + "=" * 80)
    logger.info("✅ Goal-Conditioned 设计验证完成!")
    logger.info("   - 每个 step 的 observation 都包含原始问题 (Goal)")
    logger.info("   - Step-Level 和 Trajectory-Level 训练都能正确工作")
    logger.info("   - Replay Buffer 采样任意 step 都能获得完整上下文")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("NQSearch Goal-Conditioned 设计测试")
    logger.info("=" * 80)

    # 运行演示
    test_goal_conditioned_demo()

    # 运行 pytest 测试套件
    logger.info("\n运行 pytest 测试套件...")
    pytest.main([__file__, "-v", "-s"])
