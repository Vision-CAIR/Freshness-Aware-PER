"""测试 nq_search 数据集在 SearchEnv 中的交互

测试转换后的 nq_search 数据集能否正常与 SearchEnv 交互
"""

import sys
from pathlib import Path

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


class TestNQSearchEnv:
    """测试 nq_search 数据集与 SearchEnv 的交互"""
    
    @pytest.fixture
    def env(self):
        """创建测试环境"""
        from roll.agentic.env.search import SearchEnv, SearchEnvConfig
        
        config = SearchEnvConfig(
            dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet",
            max_instances=10,
            use_mock_api=True,
            use_remote_service=False,
            disable_limiter=True,
            max_steps=5,
            max_search_calls=3
        )
        
        env = SearchEnv(config)
        yield env
        env.close()
    
    def test_env_creation(self, env):
        """测试环境创建"""
        logger.info("测试环境创建")
        assert env is not None
        assert len(env.data) == 10
        logger.info(f"✓ 环境创建成功，加载了 {len(env.data)} 个样本")
    
    def test_env_reset(self, env):
        """测试环境重置"""
        logger.info("\n测试环境重置")
        
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert isinstance(obs, str)
        assert len(obs) > 0
        
        # 检查观察是否包含系统提示词和问题
        assert "<think>" in obs or "<answer>" in obs or "Question:" in obs
        
        logger.info(f"✓ Reset 成功")
        logger.info(f"  初始观察长度: {len(obs)}")
        logger.info(f"  观察内容 (前200字符): {obs[:200]}...")
    
    def test_single_interaction(self, env):
        """测试单次交互"""
        logger.info("\n测试单次交互 (搜索 -> 回答)")
        
        # 重置环境
        obs, info = env.reset(seed=0)
        logger.info(f"问题: {obs[:200]}...")
        
        # 步骤1: 执行搜索
        action1 = '''<think>
我需要搜索相关信息来回答这个问题。
</think>

<code>
result = web_search("relevant search query")
print(result)
</code>'''
        
        obs1, reward1, terminated1, truncated1, info1 = env.step(action1)
        
        assert not terminated1, "搜索后不应该终止"
        assert not truncated1, "搜索后不应该截断"
        assert info1["action_is_valid"], "搜索动作应该有效"
        assert info1["search_calls"] == 1, "应该有1次搜索调用"
        
        logger.info(f"✓ 搜索步骤完成")
        logger.info(f"  - Reward: {reward1}")
        logger.info(f"  - Search calls: {info1['search_calls']}")
        
        # 步骤2: 提供答案
        action2 = '''<think>
根据搜索结果,我现在提供答案。
</think>

<answer>Test Answer</answer>'''
        
        obs2, reward2, terminated2, truncated2, info2 = env.step(action2)
        
        assert terminated2, "提供答案后应该终止"
        assert info2["action_is_valid"], "答案动作应该有效"
        
        logger.info(f"✓ 回答步骤完成")
        logger.info(f"  - Final reward: {reward2}")
        logger.info(f"  - Success: {info2.get('success', False)}")
        logger.info(f"  - Score: {info2.get('score', 0.0)}")
    
    def test_multiple_searches(self, env):
        """测试多次搜索"""
        logger.info("\n测试多次搜索")
        
        obs, info = env.reset(seed=1)
        
        # 执行多次搜索
        for i in range(3):
            action = f'''<think>
执行第 {i+1} 次搜索
</think>

<code>
result = web_search("query {i+1}")
print(result)
</code>'''
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert not terminated, f"第{i+1}次搜索后不应该终止"
            assert info["search_calls"] == i + 1, f"应该有{i+1}次搜索调用"
            
            logger.info(f"✓ 第{i+1}次搜索完成，累计搜索次数: {info['search_calls']}")
        
        # 达到最大搜索次数后，再次搜索应该被终止
        action = '''<code>
result = web_search("one more search")
print(result)
</code>'''
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated, "超过最大搜索次数后应该终止"
        assert info["termination_reason"] == "max_search_calls_exceeded"
        
        logger.info(f"✓ 达到最大搜索次数限制，环境正确终止")
    
    def test_invalid_action(self, env):
        """测试无效动作"""
        logger.info("\n测试无效动作")
        
        obs, info = env.reset(seed=2)
        
        # 发送无效动作（没有标签）
        invalid_action = "This is just plain text without any tags"
        
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        assert not info["action_is_valid"], "无效动作应该被标记"
        assert not terminated, "无效动作不应该立即终止环境"
        
        logger.info(f"✓ 无效动作被正确识别")
        logger.info(f"  - Observation: {obs}")
    
    def test_max_steps_truncation(self, env):
        """测试最大步数截断"""
        logger.info("\n测试最大步数截断")
        
        obs, info = env.reset(seed=3)
        
        # 执行多个思考步骤直到达到最大步数
        for i in range(env.config.max_steps):
            action = f'''<think>
这是第 {i+1} 步的思考
</think>'''
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i < env.config.max_steps - 1:
                assert not truncated, f"第{i+1}步不应该截断"
            else:
                assert truncated, "达到最大步数应该截断"
                assert info["termination_reason"] == "max_steps_reached"
                logger.info(f"✓ 达到最大步数 {env.config.max_steps}，环境正确截断")
    
    def test_different_questions(self, env):
        """测试不同问题的加载"""
        logger.info("\n测试不同问题的加载")
        
        questions = []
        
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            
            # 提取问题部分
            if "Question:" in obs:
                question = obs.split("Question:")[-1].strip()[:100]
            else:
                question = obs[:100]
            
            questions.append(question)
            logger.info(f"问题 {seed+1}: {question}...")
        
        # 验证问题不同
        unique_questions = set(questions)
        assert len(unique_questions) > 1, "应该加载不同的问题"
        
        logger.info(f"✓ 成功加载 {len(unique_questions)} 个不同的问题")
    
    def test_reward_computation(self, env):
        """测试奖励计算"""
        logger.info("\n测试奖励计算")
        
        obs, info = env.reset(seed=0)
        
        # 提取问题中的关键词作为参考
        logger.info(f"当前问题: {obs[:200]}...")
        
        # 执行一次搜索
        search_action = '''<think>
搜索相关信息
</think>

<code>
result = web_search("test query")
print(result)
</code>'''
        
        obs, reward, terminated, truncated, info = env.step(search_action)
        
        # 在 outcome-only 模式下，中间步骤的奖励应该为0
        if env.config.use_outcome_reward_only:
            assert reward == 0.0, "Outcome-only模式下中间步骤奖励应为0"
            logger.info("✓ Outcome-only模式：中间步骤奖励为0")
        
        # 提供答案
        answer_action = '''<think>
提供测试答案
</think>

<answer>Test Answer for Reward</answer>'''
        
        obs, reward, terminated, truncated, info = env.step(answer_action)
        
        assert terminated, "提供答案后应该终止"
        assert "score" in info, "Info中应该包含score"
        assert "reward" in info, "Info中应该包含reward"
        assert "reward_info" in info, "Info中应该包含reward_info"
        
        logger.info(f"✓ 奖励计算完成")
        logger.info(f"  - Score: {info['score']}")
        logger.info(f"  - Reward: {info['reward']}")
        logger.info(f"  - Reward info: {info['reward_info']}")


def test_full_workflow():
    """完整工作流程测试（不使用fixture）"""
    logger.info("\n"+"="*80)
    logger.info("完整工作流程测试")
    logger.info("="*80)
    
    from roll.agentic.env.search import SearchEnv, SearchEnvConfig
    
    # 使用转换后的完整测试集
    config = SearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/test_searchenv.parquet",
        max_instances=64,  # 完整测试集有64个样本
        use_mock_api=True,
        use_remote_service=False,
        disable_limiter=True,
        max_steps=10,
        max_search_calls=5
    )
    
    env = SearchEnv(config)
    
    logger.info(f"✓ 加载完整测试集: {len(env.data)} 个样本")
    
    # 测试几个样本
    test_count = 3
    for i in range(test_count):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试样本 {i+1}/{test_count}")
        logger.info(f"{'='*60}")
        
        obs, info = env.reset(seed=i)
        logger.info(f"问题: {obs[:150]}...")
        
        # 执行一次完整交互
        search_action = '''<think>搜索信息</think>
<code>result = web_search("test"); print(result)</code>'''
        
        obs, r, term, trunc, info = env.step(search_action)
        logger.info(f"搜索完成 - 搜索次数: {info['search_calls']}")
        
        if not (term or trunc):
            answer_action = '<think>回答</think><answer>Answer</answer>'
            obs, r, term, trunc, info = env.step(answer_action)
            logger.info(f"回答完成 - Success: {info.get('success')}, Score: {info.get('score', 0)}")
    
    env.close()
    logger.info("\n" + "="*80)
    logger.info("✅ 完整工作流程测试通过")
    logger.info("="*80)


if __name__ == "__main__":
    # 直接运行测试
    logger.info("="*80)
    logger.info("开始测试 nq_search 数据集与 SearchEnv 的交互")
    logger.info("="*80)
    
    test_full_workflow()
    
    logger.info("\n运行 pytest 测试套件...")
    pytest.main([__file__, "-v", "-s"])

