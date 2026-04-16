import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from roll.agentic.env.search import SearchEnv, SearchEnvConfig


def test_basic_search():
    """Basic search functionality test"""
    print("Testing basic search functionality...")

    config = SearchEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        max_instances=5,
        use_mock_api=False,
        use_remote_service=True,
        remote_service_url="http://172.26.104.240:30002",
        max_steps=5,
        max_search_calls=3,
        disable_limiter=True  # Disable Ray limiter for testing
    )

    env = SearchEnv(config)
    obs, info = env.reset(seed=42)

    action = '''<think>
I need to search for information.
</think>

<code>
result = web_search("test query")
print(result)
</code>'''

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Search completed. Reward: {reward}, Search calls: {info.get('search_calls', 0)}")
    return True


def test_concurrent_search(num_workers=5, num_requests=20):
    """Test concurrent search requests"""
    print(f"Testing concurrent search with {num_workers} workers, {num_requests} requests...")

    def run_single_search(seed):
        config = SearchEnvConfig(
            dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
            max_instances=10,
            use_mock_api=False,
            use_remote_service=True,
            remote_service_url="http://172.26.104.240:30002",
            max_steps=3,
            max_search_calls=2,
            disable_limiter=True
        )

        env = SearchEnv(config)
        start_time = time.time()

        try:
            obs, info = env.reset(seed=seed)

            action = f'''<think>
Testing concurrent search {seed}.
</think>

<code>
result = web_search("concurrent test {seed}")
print(result)
</code>'''

            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.time() - start_time

            return {
                "seed": seed,
                "success": True,
                "reward": reward,
                "elapsed": elapsed,
                "search_calls": info.get('search_calls', 0)
            }
        except Exception as e:
            return {
                "seed": seed,
                "success": False,
                "error": str(e),
                "elapsed": time.time() - start_time
            }

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_search, i) for i in range(num_requests)]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"Request {result['seed']}: Success, {result['elapsed']:.2f}s")
            else:
                print(f"Request {result['seed']}: Failed - {result['error']}")

    success_count = sum(1 for r in results if r["success"])
    avg_time = sum(r["elapsed"] for r in results if r["success"]) / max(success_count, 1)

    print(f"Concurrent test results: {success_count}/{num_requests} successful, avg time: {avg_time:.2f}s")
    return results


def test_trajectory_and_save(dataset_path, output_file, num_samples=10):
    """Test environment interaction and save trajectories in standard RL format: s,a,r,s,a,r,..."""
    print(f"Testing trajectory interaction with {num_samples} samples...")

    config = SearchEnvConfig(
        dataset_path=dataset_path,
        max_instances=num_samples,
        use_mock_api=False,
        use_remote_service=True,
        remote_service_url="http://172.26.104.240:30002",
        max_steps=6,
        max_search_calls=4,
        disable_limiter=True
    )

    env = SearchEnv(config)
    trajectories = []

    for i in range(num_samples):
        print(f"Processing sample {i+1}/{num_samples}")

        trajectory = {
            "sample_id": i,
            "seed": i + 42,
            "question_data": None,
            "trajectory": {},  # Clear format with step labels
            "success": False,
            "total_reward": 0.0
        }

        try:
            # obs0 - initial state
            obs0, info = env.reset(seed=i + 42)
            trajectory["question_data"] = env.current_question_data
            trajectory["trajectory"]["obs0"] = obs0

            question = env.current_question_data["question"]
            total_reward = 0.0
            step_count = 0

            # Generate actions until done
            for step in range(config.max_steps):
                # Determine action based on step and question content
                if step == 0:  # First search
                    if "railway" in question.lower():
                        search_query = "Great Western Railway Paddington railway yard slip switch"
                    elif "census" in question.lower():
                        search_query = "2020 U.S. Census miscount percentage by state"
                    elif "brainerd" in question.lower():
                        search_query = "Paul Brainerd Foundation environmental conservation"
                    else:
                        search_query = question[:50]

                    action = f'''<think>I need to search for information about this question.</think>
<code>
result = web_search("{search_query}")
print(result)
</code>'''

                elif step == 1:  # Second search
                    action = f'''<think>Let me search more specifically.</think>
<code>
result = web_search("{question[:60]}")
print(result)
</code>'''

                elif step == 2:  # Third search (if allowed)
                    action = f'''<think>One more search for specific details.</think>
<code>
result = web_search("specific details {question[:30]}")
print(result)
</code>'''

                else:  # Final answer (or forced answer if max searches reached)
                    golden_answer = env.current_question_data["golden_answers"][0] if env.current_question_data["golden_answers"] else "Unknown"
                    action = f'''<think>Based on my research, I can provide the answer.</think>
<answer>{golden_answer}</answer>'''

                # action_t
                trajectory["trajectory"][f"action{step}"] = action

                # Execute action and get reward and next state
                next_obs, reward, terminated, truncated, info = env.step(action)

                # reward_t
                trajectory["trajectory"][f"reward{step}"] = reward
                total_reward += reward

                # obs_{t+1}
                trajectory["trajectory"][f"obs{step+1}"] = next_obs
                step_count = step + 1

                # Record termination reason
                if terminated or truncated:
                    trajectory["success"] = info.get("success", False)
                    trajectory["termination_reason"] = info.get("termination_reason", "unknown")
                    break

            trajectory["total_reward"] = total_reward
            trajectory["num_steps"] = step_count

        except Exception as e:
            trajectory["error"] = str(e)
            print(f"Error in sample {i+1}: {e}")

        trajectories.append(trajectory)

    # Save trajectories
    with open(output_file, 'w') as f:
        json.dump(trajectories, f, indent=2)

    success_count = sum(1 for t in trajectories if t.get("success", False))
    avg_steps = sum(t.get("num_steps", 0) for t in trajectories) / len(trajectories)
    print(f"Trajectory test completed: {success_count}/{num_samples} successful")
    print(f"Average steps per trajectory: {avg_steps:.1f}")
    print(f"Results saved to {output_file}")

    return trajectories


def test_termination_conditions():
    """Test different termination conditions"""
    print("Testing termination conditions...")

    # Test 1: Answer termination
    print("Test 1: Answer termination")
    config = SearchEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        max_instances=1,
        max_steps=10,
        max_search_calls=5,
        disable_limiter=True
    )

    env = SearchEnv(config)
    obs, info = env.reset(seed=42)

    # Direct answer should terminate
    action = '''<think>I'll provide an answer directly.</think>
<answer>Test Answer</answer>'''

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Answer termination - Terminated: {terminated}, Reason: {info.get('termination_reason')}")

    # Test 2: Max search calls termination
    print("\nTest 2: Max search calls termination")
    config = SearchEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        max_instances=1,
        max_steps=10,
        max_search_calls=2,  # Very low limit
        disable_limiter=True
    )

    env = SearchEnv(config)
    obs, info = env.reset(seed=42)

    # First search
    action1 = '''<think>First search.</think>
<code>
result = web_search("test query 1")
print(result)
</code>'''
    obs, reward, terminated, truncated, info = env.step(action1)
    print(f"After search 1 - Search calls: {info.get('search_calls')}, Terminated: {terminated}")

    # Second search
    if not (terminated or truncated):
        action2 = '''<think>Second search.</think>
<code>
result = web_search("test query 2")
print(result)
</code>'''
        obs, reward, terminated, truncated, info = env.step(action2)
        print(f"After search 2 - Search calls: {info.get('search_calls')}, Terminated: {terminated}")

    # Third search (should be blocked)
    if not (terminated or truncated):
        action3 = '''<think>Third search.</think>
<code>
result = web_search("test query 3")
print(result)
</code>'''
        obs, reward, terminated, truncated, info = env.step(action3)
        print(f"After search 3 - Terminated: {terminated}, Reason: {info.get('termination_reason')}")

    # Test 3: Max steps truncation
    print("\nTest 3: Max steps truncation")
    config = SearchEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        max_instances=1,
        max_steps=2,  # Very low limit
        max_search_calls=5,
        disable_limiter=True
    )

    env = SearchEnv(config)
    obs, info = env.reset(seed=42)

    for step in range(3):  # Try to exceed max_steps
        action = f'''<think>Step {step+1} thinking.</think>'''
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1} - Truncated: {truncated}, Reason: {info.get('termination_reason')}")
        if terminated or truncated:
            break

    print("Termination conditions test completed.")


def test_xverify_evaluation():
    """Test xverify-based evaluation"""
    print("Testing xverify evaluation...")

    config = SearchEnvConfig(
        dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        max_instances=3,
        use_mock_api=True,
        use_remote_service=False,
        max_steps=5,
        max_search_calls=3,
        disable_limiter=True
    )

    env = SearchEnv(config)

    # Test with xverify reward computation
    try:
        from roll.agentic.reward_compute.xhpang_search_reward import compute_score

        # Test case 1: Correct answer with search
        solution_with_search = '''<think>
I need to find information about the capital of France.
</think>

<code>
result = web_search("capital of France")
print(result)
</code>

<execution_results>
Search Results:
1. Paris is the capital of France
2. France capital city information
</execution_results>

<think>
Based on the search results, the answer is clearly Paris.
</think>

<answer>Paris</answer>'''

        ground_truth = "Paris"
        question = "What is the capital of France?"

        # Test with xverify enabled
        result_xverify = compute_score(
            solution_str=solution_with_search,
            ground_truth=ground_truth,
            question=question,
            use_xverify=True,
            use_otc=False,
            return_dict=True
        )

        # Test with traditional evaluation
        result_traditional = compute_score(
            solution_str=solution_with_search,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )

        print(f"Xverify evaluation: Score={result_xverify['score']}, Method={result_xverify['evaluation_method']}")
        print(f"Traditional evaluation: Score={result_traditional['score']}, Method={result_traditional['evaluation_method']}")

        return True

    except Exception as e:
        print(f"Xverify test failed: {e}")
        return False


def test_otc_rewards():
    """Test OTC (Optimal Tool Call) rewards"""
    print("Testing OTC rewards...")

    try:
        from roll.agentic.reward_compute.xhpang_search_reward import compute_score

        # Test case: Multiple tool calls vs optimal
        solution_multiple_tools = '''<think>
I need to find information about the capital of France.
</think>

<code>
result1 = web_search("France")
print(result1)
</code>

<execution_results>
France is a country in Europe...
</execution_results>

<code>
result2 = web_search("capital of France")
print(result2)
</code>

<execution_results>
Paris is the capital of France.
</execution_results>

<code>
result3 = web_parse("https://en.wikipedia.org/wiki/Paris", "capital of France")
print(result3)
</code>

<execution_results>
Paris is the capital and most populous city of France.
</execution_results>

<answer>Paris</answer>'''

        solution_optimal = '''<think>
I need to find the capital of France.
</think>

<code>
result = web_search("capital of France")
print(result)
</code>

<execution_results>
Paris is the capital of France.
</execution_results>

<answer>Paris</answer>'''

        ground_truth = "Paris"
        question = "What is the capital of France?"

        # Test OTC-PPO (doesn't need correct trajectories)
        result_multiple_ppo = compute_score(
            solution_str=solution_multiple_tools,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        result_optimal_ppo = compute_score(
            solution_str=solution_optimal,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        # Test OTC-GRPO (needs correct trajectories)
        correct_trajectories = [solution_optimal, "Paris"]  # Optimal trajectory

        result_multiple_grpo = compute_score(
            solution_str=solution_multiple_tools,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="grpo",
            correct_trajectories=correct_trajectories,
            return_dict=True
        )

        print(f"Multiple tools (PPO): Score={result_multiple_ppo['score']:.3f}")
        print(f"Optimal tools (PPO): Score={result_optimal_ppo['score']:.3f}")
        print(f"Multiple tools (GRPO): Score={result_multiple_grpo['score']:.3f}")

        # OTC should penalize excessive tool usage
        assert result_optimal_ppo['score'] >= result_multiple_ppo['score'], "OTC should favor optimal tool usage"

        return True

    except Exception as e:
        print(f"OTC test failed: {e}")
        return False


def test_combined_xverify_otc():
    """Test combined xverify and OTC evaluation"""
    print("Testing combined xverify + OTC evaluation...")

    try:
        from roll.agentic.reward_compute.xhpang_search_reward import compute_score

        solution_str = '''<think>
I need to search for information about the largest city in the US.
</think>

<code>
result = web_search("largest city United States population")
print(result)
</code>

<execution_results>
New York City is the most populous city in the United States.
</execution_results>

<answer>New York City</answer>'''

        ground_truth = {"target": ["New York City", "NYC", "New York"]}
        question = "What is the largest city in the United States?"

        # Test all combinations
        result_basic = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=False,
            return_dict=True
        )

        result_xverify_only = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=True,
            use_otc=False,
            return_dict=True
        )

        result_otc_only = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=False,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        result_combined = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            use_xverify=True,
            use_otc=True,
            otc_method="ppo",
            return_dict=True
        )

        print(f"Basic: Score={result_basic['score']:.3f}, Method={result_basic['evaluation_method']}")
        print(f"Xverify only: Score={result_xverify_only['score']:.3f}, Method={result_xverify_only['evaluation_method']}")
        print(f"OTC only: Score={result_otc_only['score']:.3f}")
        print(f"Combined: Score={result_combined['score']:.3f}, Method={result_combined['evaluation_method']}")

        return True

    except Exception as e:
        print(f"Combined test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        # Test 1: Basic functionality
        test_basic_search()

        # Test 2: Termination conditions
        test_termination_conditions()

        # Test 3: Concurrent requests
        test_concurrent_search(num_workers=3, num_requests=10)

        # Test 4: Train dataset trajectories
        train_path =  "/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet"
        test_trajectory_and_save(train_path, "train_trajectories.json", num_samples=5)

        # Test 5: Validation dataset trajectories
        val_path = "/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet"
        test_trajectory_and_save(val_path, "val_trajectories.json", num_samples=3)

        # Test 6: Advanced xverify evaluation
        test_xverify_evaluation()

        # Test 7: OTC rewards
        test_otc_rewards()

        # Test 8: Combined xverify + OTC
        test_combined_xverify_otc()

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
