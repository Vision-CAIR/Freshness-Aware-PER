from dataclasses import dataclass, field
from typing import Optional, List, Dict
from roll.agentic.env.base import BaseEnvConfig


@dataclass
class SearchEnvConfig(BaseEnvConfig):
    """Configuration for Search environment"""
    
    # Dataset config
    dataset_path: str = "/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
    max_instances: int = 1000
    
    # Search API config
    search_api_url: str = "https://api.example.com/search"
    parse_api_url: str = "https://api.example.com/parse"
    img_parse_api_url: str = "https://api.example.com/parse_img"
    api_key: Optional[str] = None
    use_mock_api: bool = False  # Set to True for testing only

    # Remote service config (for HTTP-based tool execution)
    use_remote_service: bool = True  # Use remote HTTP service by default
    remote_service_url: str = "http://172.26.104.240:30002"  # Default tool server URL
    remote_service_timeout: int = 30  # Timeout in seconds
    
    # Rate limiting
    max_concurrent_calls: int = 5
    limiter_tag: str = "search_env"
    disable_limiter: bool = False  # Set to True to disable Ray-based limiter for testing
    
    # Environment settings
    max_steps: int = 10
    max_search_calls: int = 5
    
    # Patterns for parsing actions
    think_pattern: str = r"<think>(.*?)</think>"
    code_pattern: str = r"<code>(.*?)</code>"
    answer_pattern: str = r"<answer>(.*?)</answer>"
    
    # Reward settings - Outcome-based only
    use_outcome_reward_only: bool = True  # Only use final outcome reward, no step penalties
    correct_answer_reward: float = 1.0
    step_penalty: float = 0.0  # Disabled when use_outcome_reward_only=True
    invalid_action_penalty: float = 0.0  # Disabled when use_outcome_reward_only=True

    # Advanced reward computation
    use_xverify: bool = True  # Use xverify for final reward computation
    use_otc: bool = False  # Use OTC (Optimal Tool Call) rewards
    otc_method: str = "ppo"  # "ppo" or "grpo"
    otc_alpha: float = 1.0  # OTC weight
    otc_c: float = 1.0  # OTC smoothing constant
    xverify_config: Optional[Dict] = None  # Custom xverify configuration
    
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<code>", "</code>", 
                                "<answer>", "</answer>", "<execution_results>", "</execution_results>"]
    )