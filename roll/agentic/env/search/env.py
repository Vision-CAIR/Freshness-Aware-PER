import random
from typing import Dict, Any, Tuple
import datasets
from roll.agentic.env.base import BaseEnv
from roll.agentic.utils import all_seed
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from .config import SearchEnvConfig
from .utils import execute_search_code, parse_action_content


class SearchEnv(BaseEnv):
    """
    Search environment for question answering with web search capabilities.
    """

    def __init__(self, config: SearchEnvConfig = None):
        self.config = config or SearchEnvConfig()
        super(SearchEnv, self).__init__(self.config)
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Initialize rate limiter (skip if testing)
        if hasattr(self.config, 'disable_limiter') and self.config.disable_limiter:
            self.limiter = None
        else:
            self.limiter = get_global_limiter(
                tag=self.config.limiter_tag,
                max_concurrent_calls=self.config.max_concurrent_calls
            )
        
        # Environment state
        self.current_question_idx = None
        self.current_question_data = None
        self.step_count = 0
        self.search_call_count = 0
        self.render_cache = None
        
    def _load_dataset(self):
        """Load and filter dataset"""
        df = datasets.load_dataset("parquet", data_files=self.config.dataset_path)["train"]
        if self.config.max_instances > 0:
            df = df.select(range(min(self.config.max_instances, len(df))))
        return df

    def reset(self, seed=None, **kwargs):
        """Reset environment and return initial observation"""
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(self.data) - 1)

        self.current_question_data = self.data[self.current_question_idx]
        self.step_count = 0
        self.search_call_count = 0

        # Create initial observation from prompt
        system_prompt = self.current_question_data["prompt"][0]["content"]
        user_question = self.current_question_data["prompt"][1]["content"]

        initial_obs = f"{system_prompt}\n\nQuestion: {user_question}"
        self.render_cache = initial_obs

        return initial_obs, {}

    def step(self, action: str):
        """Execute one step in the environment"""
        self.step_count += 1

        # Parse action content
        action_info = self._parse_action(action)

        if not action_info["is_valid"]:
            # Use outcome-only reward: no penalty for invalid actions during episode
            reward = 0.0 if self.config.use_outcome_reward_only else self.config.invalid_action_penalty
            obs = "Invalid action format. Please use proper <think>, <code>, or <answer> tags."
            terminated = False
            truncated = False
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "step_count": self.step_count,
                "search_calls": self.search_call_count
            }
            self.render_cache = obs
            return obs, reward, terminated, truncated, info

        # Check if answer is provided - this terminates the episode
        if action_info["answer"]:
            reward, success, score, reward_info = self._compute_final_reward(action_info["answer"])
            obs = f"Final answer submitted: {action_info['answer']}"
            terminated = True
            truncated = False
            info = {
                "action_is_effective": True,
                "action_is_valid": True,
                "success": success,
                "score": score,  # Raw correctness score (0-1)
                "reward": reward,  # Final reward after OTC/other adjustments
                "reward_info": reward_info,  # Detailed reward breakdown
                "step_count": self.step_count,
                "search_calls": self.search_call_count,
                "termination_reason": "answer_provided"
            }
            self.render_cache = obs
            return obs, reward, terminated, truncated, info

        # Execute code if present
        if action_info["code"]:
            # Check if we've reached max search calls before executing
            if self.search_call_count >= self.config.max_search_calls:
                obs = "<execution_results>\nError: Maximum search calls exceeded. Please provide your final answer.\n</execution_results>"
                # Use outcome-only reward: no penalty during episode
                reward = 0.0 if self.config.use_outcome_reward_only else self.config.invalid_action_penalty
                terminated = True  # Terminate when max search calls reached
                truncated = False
                info = {
                    "action_is_effective": False,
                    "action_is_valid": True,
                    "success": False,
                    "score": 0.0,  # Failed to provide answer
                    "reward": reward,
                    "step_count": self.step_count,
                    "search_calls": self.search_call_count,
                    "termination_reason": "max_search_calls_exceeded"
                }
                self.render_cache = obs
                return obs, reward, terminated, truncated, info
            else:
                execution_result = self._execute_code(action_info["code"])
                obs = f"<execution_results>\n{execution_result}\n</execution_results>"
                # Use outcome-only reward: no step penalty
                reward = 0.0 if self.config.use_outcome_reward_only else self.config.step_penalty
                terminated = False
                truncated = False
        else:
            # Just thinking, no code execution
            obs = "Continue with your reasoning..."
            # Use outcome-only reward: no step penalty
            reward = 0.0 if self.config.use_outcome_reward_only else self.config.step_penalty
            terminated = False
            truncated = False

        # Check max steps - this should truncate, not terminate
        if self.step_count >= self.config.max_steps:
            terminated = False
            truncated = True
            # Use outcome-only reward: no penalty for timeout
            if not self.config.use_outcome_reward_only:
                reward += self.config.invalid_action_penalty

        info = {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": False,
            "step_count": self.step_count,
            "search_calls": self.search_call_count,
            "termination_reason": "max_steps_reached" if truncated else None
        }

        self.render_cache = obs
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: str) -> Dict[str, Any]:
        """Parse action to extract think, code, and answer components"""
        return parse_action_content(action, self.config)

    def _execute_code(self, code: str) -> str:
        """Execute search code with rate limiting"""
        try:
            # Acquire rate limit permission if limiter is available
            acquire_id = None
            if self.limiter:
                acquire_id = self.limiter.acquire()

            try:
                result = execute_search_code(
                    code,
                    self.config,
                    self._increment_search_count
                )
                return result
            finally:
                # Release rate limit permission if acquired
                if self.limiter and acquire_id:
                    self.limiter.release(acquire_id)

        except Exception as e:
            return f"Execution error: {str(e)}"

    def _increment_search_count(self):
        """Increment search call counter"""
        self.search_call_count += 1

    def _compute_final_reward(self, answer: str) -> Tuple[float, bool, float, Dict]:
        """
        Compute final reward for answer using advanced reward computation
        Returns: (final_reward, success, raw_score, reward_info)
        """
        golden_answers = self.current_question_data["golden_answers"]
        question = self.current_question_data.get("question", "")

        try:
            from roll.agentic.reward_compute.xhpang_search_reward import compute_score

            # Prepare solution string with answer
            solution_str = f"<answer>{answer}</answer>"

            # Use advanced reward computation
            result = compute_score(
                solution_str=solution_str,
                ground_truth=golden_answers,
                question=question,
                use_xverify=self.config.use_xverify,
                use_otc=self.config.use_otc,
                otc_method=self.config.otc_method,
                otc_alpha=self.config.otc_alpha,
                otc_c=self.config.otc_c,
                xverify_config=self.config.xverify_config,
                return_dict=True
            )

            raw_score = result["score"]  # Raw correctness score (0-1)
            final_reward = raw_score * self.config.correct_answer_reward
            success = result["is_correct"]

            reward_info = {
                "evaluation_method": result.get("evaluation_method", "unknown"),
                "raw_score": raw_score,
                "base_reward": final_reward,
                "otc_info": result.get("otc_info", {}),
                "xverify_info": result.get("xverify_info", {})
            }

            return final_reward, success, raw_score, reward_info

        except Exception as e:
            # Fallback to simple matching if advanced reward fails
            print(f"Advanced reward computation failed: {e}, falling back to simple matching")
            answer_clean = answer.strip().lower()
            for golden in golden_answers:
                if golden.strip().lower() in answer_clean or answer_clean in golden.strip().lower():
                    reward_info = {"evaluation_method": "simple_matching", "raw_score": 1.0, "base_reward": self.config.correct_answer_reward}
                    return self.config.correct_answer_reward, True, 1.0, reward_info

            reward_info = {"evaluation_method": "simple_matching", "raw_score": 0.0, "base_reward": 0.0}
            return 0.0, False, 0.0, reward_info

    def render(self, mode: str = "text"):
        """Render current state"""
        return self.render_cache

    def close(self):
        """Clean up resources"""
        pass


if __name__ == "__main__":
    # Add project root to Python path for standalone execution
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    sys.path.insert(0, project_root)

    # Simple demonstration of search environment
    print("Search Environment Demo")
    print("Expected trajectory: observation0 → action0 → observation1 → action1 → observation2")

    config = SearchEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        max_instances=5,
        disable_limiter=True  # Disable for standalone demo
    )

    env = SearchEnv(config)

    # Reset and get initial question
    obs0, info = env.reset(seed=42)
    print(f"\nInitial question: {obs0[:200]}...")

    # Search action
    action0 = '''<think>I need to search for information.</think>
<code>
result = web_search("search query")
print(result)
</code>'''

    obs1, reward1, terminated1, truncated1, info1 = env.step(action0)
    print(f"\nSearch completed. Reward: {reward1}")

    # Answer action
    if not (terminated1 or truncated1):
        action1 = '''<think>Based on results.</think>
<answer>Sample answer</answer>'''

        obs2, reward2, terminated2, truncated2, info2 = env.step(action1)
        print(f"Answer submitted. Final reward: {reward2}, Success: {info2.get('success', False)}")

    print("Demo completed.")