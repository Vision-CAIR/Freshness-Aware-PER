"""测试完整的 NQSearchEnv 环境

测试新创建的 NQSearchEnv 与检索服务器的集成
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
    """测试 NQSearchEnv 的完整功能"""
    
    @pytest.fixture
    def env(self):
        """创建测试环境"""
        from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig
        
        config = NQSearchEnvConfig(
            dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet",
            max_instances=10,
            retrieval_server_url="http://127.0.0.1:8100/retrieve",
            retrieval_topk=3,
            disable_limiter=True,
            max_steps=5,
            max_search_calls=3
        )
        
        env = NQSearchEnv(config)
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
        
        # 检查观察是否包含预期内容
        assert "Question:" in obs or "<search>" in obs or "<answer>" in obs
        
        logger.info(f"✓ Reset 成功")
        logger.info(f"  初始观察长度: {len(obs)}")
    
    def test_search_action(self, env):
        """测试搜索动作"""
        logger.info("\n测试搜索动作")
        
        obs, info = env.reset(seed=0)
        logger.info(f"问题: {obs[:150]}...")
        
        # 执行搜索
        search_action = '''<think>
I need to search for information about this question.
</think>

<search>owner of reading football club</search>'''
        
        obs, reward, terminated, truncated, info = env.step(search_action)
        
        assert not terminated, "搜索后不应该终止"
        assert info["action_is_valid"], "搜索动作应该有效"
        assert info["search_calls"] == 1, "应该有1次搜索调用"
        assert "<information>" in obs, "返回结果应该包含<information>标签"
        
        logger.info(f"✓ 搜索执行成功")
        logger.info(f"  - Search calls: {info['search_calls']}")
        logger.info(f"  - Result preview: {obs[:200]}...")
    
    def test_answer_action(self, env):
        """测试回答动作"""
        logger.info("\n测试回答动作")
        
        obs, info = env.reset(seed=0)
        
        # 先搜索
        search_action = '''<think>Search first</think>
<search>test query</search>'''
        env.step(search_action)
        
        # 然后回答
        answer_action = '''<think>
Based on information, I can answer.
</think>

<answer>Test Answer</answer>'''
        
        obs, reward, terminated, truncated, info = env.step(answer_action)
        
        assert terminated, "提供答案后应该终止"
        assert info["action_is_valid"], "答案动作应该有效"
        assert "score" in info, "Info中应该包含score"
        
        logger.info(f"✓ 回答执行成功")
        logger.info(f"  - Reward: {reward}")
        logger.info(f"  - Success: {info.get('success', False)}")
        logger.info(f"  - Score: {info.get('score', 0.0)}")
    
    def test_em_evaluation(self, env):
        """测试 EM 评估"""
        logger.info("\n测试 EM 评估")
        
        # 使用已知答案的问题
        obs, info = env.reset(seed=2)  # "who got the first nobel prize in physics?"
        
        # 执行搜索
        search_action = '<think>Search</think><search>first nobel prize physics</search>'
        env.step(search_action)
        
        # 提供正确答案
        correct_answer = '<think>Answer</think><answer>Wilhelm Conrad Röntgen</answer>'
        obs, reward, terminated, truncated, info = env.step(correct_answer)
        
        logger.info(f"正确答案测试:")
        logger.info(f"  - Predicted: Wilhelm Conrad Röntgen")
        logger.info(f"  - Golden: {env.current_question_data['golden_answers']}")
        logger.info(f"  - Success: {info.get('success', False)}")
        logger.info(f"  - Score: {info.get('score', 0.0)}")
        logger.info(f"  - Reward: {reward}")
    
    def test_max_search_calls(self, env):
        """测试最大搜索次数限制"""
        logger.info("\n测试最大搜索次数限制")
        
        obs, info = env.reset(seed=1)
        
        # 执行多次搜索
        for i in range(3):
            action = f'<search>query {i+1}</search>'
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not terminated:
                logger.info(f"✓ 第{i+1}次搜索成功，累计: {info['search_calls']}")
        
        # 尝试超过限制
        action = '<search>one more search</search>'
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated, "超过最大搜索次数应该终止"
        assert info["termination_reason"] == "max_search_calls_exceeded"
        
        logger.info(f"✓ 达到最大搜索次数限制，环境正确终止")
    
    def test_max_steps_truncation(self, env):
        """测试最大步数截断"""
        logger.info("\n测试最大步数截断")
        
        obs, info = env.reset(seed=3)
        
        # 执行多个思考步骤
        for i in range(env.config.max_steps):
            action = f'<think>Step {i+1}</think>'
            obs, reward, terminated, truncated, info = env.step(action)
            
            if truncated:
                assert info["termination_reason"] == "max_steps_reached"
                logger.info(f"✓ 达到最大步数 {env.config.max_steps}，环境正确截断")
                break


def test_full_workflow():
    """完整工作流程测试"""
    logger.info("\n"+"="*80)
    logger.info("完整工作流程测试")
    logger.info("="*80)
    
    from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig
    
    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/test_searchenv.parquet",
        max_instances=64,
        retrieval_server_url="http://127.0.0.1:8100/retrieve",
        retrieval_topk=3,
        disable_limiter=True,
        max_steps=10,
        max_search_calls=5
    )
    
    env = NQSearchEnv(config)
    
    logger.info(f"✓ 加载完整测试集: {len(env.data)} 个样本")
    
    # 测试几个样本
    test_count = 3
    for i in range(test_count):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试样本 {i+1}/{test_count}")
        logger.info(f"{'='*60}")
        
        obs, info = env.reset(seed=i)
        logger.info(f"问题 (前100字符): {obs[:100]}...")
        
        # 执行搜索
        search_action = '''<think>Need to search</think>
<search>test query</search>'''
        
        obs, r, term, trunc, info = env.step(search_action)
        logger.info(f"搜索完成 - 搜索次数: {info['search_calls']}")
        logger.info(f"结果预览: {obs[:150]}...")
        
        # 提供答案
        if not (term or trunc):
            answer_action = '<think>Answer based on results</think><answer>Test Answer</answer>'
            obs, r, term, trunc, info = env.step(answer_action)
            logger.info(f"回答完成 - Success: {info.get('success')}, Score: {info.get('score', 0)}")
    
    env.close()
    logger.info("\n" + "="*80)
    logger.info("✅ 完整工作流程测试通过")
    logger.info("="*80)


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("开始测试 NQSearchEnv")
    logger.info("="*80)
    
    test_full_workflow()
    
    logger.info("\n运行 pytest 测试套件...")
    pytest.main([__file__, "-v", "-s"])

