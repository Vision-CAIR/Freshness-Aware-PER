"""NQ Search Environment

基于检索增强的问答环境，使用 Natural Questions 数据集和专用检索服务器
"""

import random
from typing import Dict, Any, Tuple
import datasets
from roll.agentic.env.base import BaseEnv
from roll.agentic.utils import all_seed
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from roll.utils.logging import get_logger
from .config import NQSearchEnvConfig
from .utils import parse_action_content, call_retrieval_server, evaluate_answer_em

logger = get_logger()


class NQSearchEnv(BaseEnv):
    """NQ Search 环境
    
    支持使用 <search> 标签进行检索增强的问答任务
    """

    def __init__(self, config: NQSearchEnvConfig = None):
        """初始化环境
        
        Args:
            config: 环境配置对象
        """
        self.config = config or NQSearchEnvConfig()
        super(NQSearchEnv, self).__init__(self.config)
        
        # 加载数据集
        self.data = self._load_dataset()
        logger.info(f"Loaded {len(self.data)} instances from dataset")
        
        # 初始化限流器
        if hasattr(self.config, 'disable_limiter') and self.config.disable_limiter:
            self.limiter = None
        else:
            self.limiter = get_global_limiter(
                tag=self.config.limiter_tag,
                max_concurrent_calls=self.config.max_concurrent_calls
            )
        
        # 环境状态
        self.current_question_idx = None
        self.current_question_data = None
        self.step_count = 0
        self.search_call_count = 0
        self.render_cache = None

        # Goal-Conditioned RL: 存储当前问题，确保每个step都能访问
        self.current_question = None
        self.system_prompt = None
        
    def _load_dataset(self):
        """加载并过滤数据集"""
        df = datasets.load_dataset("parquet", data_files=self.config.dataset_path)["train"]
        if self.config.max_instances > 0:
            df = df.select(range(min(self.config.max_instances, len(df))))
        return df

    def reset(self, seed=None, **kwargs):
        """重置环境并返回初始观察
        
        Args:
            seed: 随机种子
            
        Returns:
            (observation, info) 元组
        """
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(self.data) - 1)

        self.current_question_data = self.data[self.current_question_idx]
        self.step_count = 0
        self.search_call_count = 0

        # 从 prompt 创建初始观察
        # NQ数据集格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        self.system_prompt = self.current_question_data["prompt"][0]["content"]
        self.current_question = self.current_question_data["prompt"][1]["content"]

        # Goal-Conditioned: 初始observation包含系统提示和问题
        # 使用空的state_info表示这是初始状态，等待第一次动作
        initial_obs = self._build_observation(
            state_info="Please analyze the question and decide whether to search for information or provide an answer.",
            include_system_prompt=True
        )
        self.render_cache = initial_obs

        return initial_obs, {}

    def step(self, action: str):
        """执行一步动作
        
        Args:
            action: 动作字符串（包含 <think>, <search>, <answer> 标签）
            
        Returns:
            (observation, reward, terminated, truncated, info) 元组
        """
        self.step_count += 1

        # 解析动作内容
        action_info = self._parse_action(action)

        if not action_info["is_valid"]:
            # 无效动作处理
            reward = 0.0 if self.config.use_outcome_reward_only else self.config.invalid_action_penalty
            # Goal-Conditioned: 即使无效动作，也要包含Goal
            obs = self._build_observation(
                "Invalid action format. Please use proper <think>, <search>, or <answer> tags."
            )
            terminated = False
            truncated = False
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
            }
            info = {
                "metrics": metrics,
                "step_count": self.step_count,
                "search_calls": self.search_call_count
            }
            self.render_cache = obs
            return obs, reward, terminated, truncated, info

        # 检查是否提供了答案 - 这将终止回合
        if action_info["answer"]:
            reward, success, score, reward_info = self._compute_final_reward(action_info["answer"])
            # Goal-Conditioned: 终止时也包含Goal（虽然episode结束，但保持一致性）
            obs = self._build_observation(
                f"Final answer submitted: {action_info['answer']}"
            )
            terminated = True
            truncated = False
            metrics = {
                "action_is_effective": True,
                "action_is_valid": True,
                "success": success,
            }
            info = {
                "metrics": metrics,
                "score": score,
                "reward": reward,
                "reward_info": reward_info,
                "step_count": self.step_count,
                "search_calls": self.search_call_count,
                "termination_reason": "answer_provided"
            }
            self.render_cache = obs
            return obs, reward, terminated, truncated, info

        # 如果有搜索查询，执行检索
        if action_info["search_query"]:
            # 检查是否超过最大搜索次数
            if self.search_call_count >= self.config.max_search_calls:
                # Goal-Conditioned: 超出搜索次数时也包含Goal
                obs = self._build_observation(
                    "<information>\nError: Maximum search calls exceeded. Please provide your final answer.\n</information>"
                )
                reward = 0.0 if self.config.use_outcome_reward_only else self.config.invalid_action_penalty
                terminated = True
                truncated = False
                metrics = {
                    "action_is_effective": False,
                    "action_is_valid": True,
                    "success": False,
                }
                info = {
                    "metrics": metrics,
                    "score": 0.0,
                    "reward": reward,
                    "step_count": self.step_count,
                    "search_calls": self.search_call_count,
                    "termination_reason": "max_search_calls_exceeded"
                }
                self.render_cache = obs
                return obs, reward, terminated, truncated, info
            else:
                # 执行检索
                search_result = self._execute_search(action_info["search_query"])
                # Goal-Conditioned: 搜索结果observation包含Goal
                obs = self._build_observation(
                    f"<information>\n{search_result}\n</information>"
                )
                reward = 0.0 if self.config.use_outcome_reward_only else self.config.step_penalty
                terminated = False
                truncated = False
        else:
            # 只有思考，没有搜索或答案
            # Goal-Conditioned: 继续推理时也包含Goal
            obs = self._build_observation("Continue with your reasoning...")
            reward = 0.0 if self.config.use_outcome_reward_only else self.config.step_penalty
            terminated = False
            truncated = False

        # 检查最大步数
        if self.step_count >= self.config.max_steps:
            terminated = False
            truncated = True
            if not self.config.use_outcome_reward_only:
                reward += self.config.invalid_action_penalty

        metrics = {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": False,
        }
        info = {
            "metrics": metrics,
            "step_count": self.step_count,
            "search_calls": self.search_call_count,
            "termination_reason": "max_steps_reached" if truncated else None
        }

        self.render_cache = obs
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: str) -> Dict[str, Any]:
        """解析动作内容"""
        return parse_action_content(action, self.config)

    def _build_observation(self, state_info: str, include_system_prompt: bool = True) -> str:
        """构建包含Goal的observation

        Goal-Conditioned RL设计：每个step的observation都应该包含Goal（原始问题），
        确保无论是Step-Level还是Trajectory-Level训练，模型都能清楚地知道要解决什么问题。

        Args:
            state_info: 当前状态信息（搜索结果、错误信息等）
            include_system_prompt: 是否包含系统提示（任务说明）

        Returns:
            包含Goal和状态信息的完整observation
        """
        # 格式：[系统提示] + Goal（问题）+ 当前状态
        if include_system_prompt and self.system_prompt:
            return f"{self.system_prompt}\n\nQuestion: {self.current_question}\n\n{state_info}"
        else:
            return f"Question: {self.current_question}\n\n{state_info}"

    def _execute_search(self, query: str) -> str:
        """执行搜索并返回结果
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            格式化的搜索结果
        """
        try:
            # 获取限流许可
            acquire_id = None
            if self.limiter:
                acquire_id = self.limiter.acquire()

            try:
                result = call_retrieval_server(
                    query,
                    self.config,
                    self._increment_search_count
                )
                return result
            finally:
                # 释放限流许可
                if self.limiter and acquire_id:
                    self.limiter.release(acquire_id)

        except Exception as e:
            logger.error(f"Search execution error: {e}")
            return f"Search error: {str(e)}"

    def _increment_search_count(self):
        """增加搜索调用计数"""
        self.search_call_count += 1

    def _compute_final_reward(self, answer: str) -> Tuple[float, bool, float, Dict]:
        """计算最终奖励
        
        Args:
            answer: 模型预测的答案
            
        Returns:
            (final_reward, success, raw_score, reward_info) 元组
        """
        golden_answers = self.current_question_data["golden_answers"]
        question = self.current_question_data.get("question", "")

        try:
            if self.config.use_em_evaluation:
                # 使用 EM 评估
                success, raw_score = evaluate_answer_em(
                    predicted_answer=answer,
                    golden_answers=golden_answers,
                    ignore_case=self.config.em_ignore_case,
                    ignore_punctuation=self.config.em_ignore_punctuation,
                    ignore_articles=self.config.em_ignore_articles
                )
                
                final_reward = raw_score * self.config.correct_answer_reward
                
                reward_info = {
                    "evaluation_method": "exact_match",
                    "raw_score": raw_score,
                    "base_reward": final_reward,
                    "predicted_answer": answer,
                    "golden_answers": golden_answers
                }
                
                logger.debug(f"EM Evaluation: success={success}, score={raw_score}")
                
                return final_reward, success, raw_score, reward_info
            else:
                # 简单字符串匹配作为后备方案
                answer_clean = answer.strip().lower()
                for golden in golden_answers:
                    if golden.strip().lower() in answer_clean or answer_clean in golden.strip().lower():
                        reward_info = {
                            "evaluation_method": "simple_matching",
                            "raw_score": 1.0,
                            "base_reward": self.config.correct_answer_reward
                        }
                        return self.config.correct_answer_reward, True, 1.0, reward_info

                reward_info = {
                    "evaluation_method": "simple_matching",
                    "raw_score": 0.0,
                    "base_reward": 0.0
                }
                return 0.0, False, 0.0, reward_info

        except Exception as e:
            logger.error(f"Reward computation error: {e}")
            # 出错时返回零奖励
            reward_info = {
                "evaluation_method": "error",
                "raw_score": 0.0,
                "base_reward": 0.0,
                "error": str(e)
            }
            return 0.0, False, 0.0, reward_info

    def render(self, mode: str = "text"):
        """渲染当前状态"""
        return self.render_cache

    def close(self):
        """清理资源"""
        pass


if __name__ == "__main__":
    # 独立运行示例
    import sys
    import os
    import logging
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    sys.path.insert(0, project_root)

    logging.basicConfig(level=logging.INFO)

    print("NQ Search Environment Demo (Goal-Conditioned)")
    print("=" * 80)

    config = NQSearchEnvConfig(
        dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet",
        max_instances=5,
        disable_limiter=True
    )

    env = NQSearchEnv(config)

    # 重置并获取初始问题
    obs0, info = env.reset(seed=42)
    print(f"\n[Step 0] Initial observation (包含Goal):")
    print("-" * 40)
    print(f"{obs0[:500]}...")
    print("-" * 40)

    # 搜索动作
    action0 = '''<think>I need to search for information about this question.</think>
<search>nobel prize physics first winner</search>'''

    print(f"\n[Action 0]: {action0}\n")
    obs1, reward1, terminated1, truncated1, info1 = env.step(action0)
    print(f"[Step 1] Observation (注意：每个step都包含Goal!):")
    print("-" * 40)
    print(f"{obs1[:600]}...")
    print("-" * 40)
    print(f"Reward: {reward1}, Terminated: {terminated1}\n")

    # 答案动作
    if not (terminated1 or truncated1):
        action1 = '''<think>Based on the search results, I can provide an answer.</think>
<answer>Wilhelm Conrad Röntgen</answer>'''

        print(f"[Action 1]: {action1}\n")
        obs2, reward2, terminated2, truncated2, info2 = env.step(action1)
        print(f"[Step 2] Final observation (终止时也包含Goal):")
        print("-" * 40)
        print(f"{obs2[:400]}...")
        print("-" * 40)
        print(f"Final reward: {reward2}, Success: {info2.get('success', False)}")
        print(f"Score: {info2.get('score', 0.0)}")

    print("\n" + "=" * 80)
    print("Goal-Conditioned设计验证完成!")
    print("每个step的observation都包含了原始问题(Goal)，")
    print("确保Step-Level和Trajectory-Level训练都能正确学习。")
    print("=" * 80)

