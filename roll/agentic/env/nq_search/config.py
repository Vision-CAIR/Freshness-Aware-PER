"""NQ Search Environment Configuration

配置用于 Natural Questions (NQ) 搜索环境，使用专用的检索服务器
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from roll.agentic.env.base import BaseEnvConfig


@dataclass
class NQSearchEnvConfig(BaseEnvConfig):
    """NQ Search 环境配置"""
    
    # 数据集配置
    dataset_path: str = "/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet"
    max_instances: int = 1000
    
    # 检索服务器配置
    retrieval_server_url: str = "http://127.0.0.1:8100/retrieve"  # 检索服务器地址
    retrieval_timeout: int = 30  # 超时时间（秒）
    retrieval_topk: int = 3  # 返回前k个相关文档
    return_scores: bool = True  # 是否返回相似度分数
    
    # 限流配置
    max_concurrent_calls: int = 5
    limiter_tag: str = "nq_search_env"
    disable_limiter: bool = False
    
    # 环境设置
    max_steps: int = 10
    max_search_calls: int = 5
    
    # 动作解析模式
    think_pattern: str = r"<think>(.*?)</think>"
    search_pattern: str = r"<search>(.*?)</search>"  # NQ数据集使用<search>标签
    answer_pattern: str = r"<answer>(.*?)</answer>"
    
    # 奖励设置 - 只使用最终结果奖励
    use_outcome_reward_only: bool = True
    correct_answer_reward: float = 1.0
    step_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    
    # 奖励计算配置
    use_em_evaluation: bool = True  # 使用精确匹配（Exact Match）评估
    em_ignore_case: bool = True  # EM评估时忽略大小写
    em_ignore_punctuation: bool = True  # EM评估时忽略标点符号
    em_ignore_articles: bool = True  # EM评估时忽略冠词(a, an, the)
    
    # 特殊标记列表
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "<think>", "</think>", 
            "<search>", "</search>", 
            "<answer>", "</answer>", 
            "<information>", "</information>"
        ]
    )

