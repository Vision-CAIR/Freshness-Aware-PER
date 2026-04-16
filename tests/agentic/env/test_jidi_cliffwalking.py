"""
Test script for Jidi CliffWalking environment in ROLL framework

This test verifies that the Jidi CliffWalking environment is properly
integrated with ROLL and can be used for training.
"""

import pytest
import sys
import os
from pathlib import Path

# Add ROLL project root to path
roll_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(roll_root))

from roll.agentic.env.jidi.config import CliffWalkingConfig
from roll.agentic.env.jidi.env import CliffWalkingEnv
from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS


def test_jidi_cliffwalking_registration():
    """Test that Jidi CliffWalking is properly registered in ROLL"""
    assert "jidi_cliffwalking" in REGISTERED_ENVS, "CliffWalkingEnv not registered"
    assert "jidi_cliffwalking" in REGISTERED_ENV_CONFIGS, "CliffWalkingConfig not registered"
    
    # Verify the registered classes
    assert REGISTERED_ENVS["jidi_cliffwalking"] == CliffWalkingEnv
    assert REGISTERED_ENV_CONFIGS["jidi_cliffwalking"] == CliffWalkingConfig


def test_cliffwalking_config():
    """Test CliffWalking configuration"""
    config = CliffWalkingConfig()
    
    # Test default values
    assert config.jidi_env_name == "cliffwalking"
    assert config.max_steps == 200
    assert config.max_tokens_per_step == 64
    assert config.grid_height == 4
    assert config.grid_width == 12
    
    # Test instruction is generated
    assert len(config.env_instruction) > 0
    assert "cliff" in config.env_instruction.lower()
    
    # Test custom values
    custom_config = CliffWalkingConfig(
        max_steps=100,
        max_tokens_per_step=32,
        format_penalty=-5.0
    )
    assert custom_config.max_steps == 100
    assert custom_config.max_tokens_per_step == 32
    assert custom_config.format_penalty == -5.0


def test_cliffwalking_env_creation():
    """Test creating CliffWalking environment"""
    config = CliffWalkingConfig(max_steps=50)
    
    try:
        env = CliffWalkingEnv(config)
        assert env is not None
        assert env.config == config
        assert hasattr(env, 'jidi_env')
        assert hasattr(env, 'interactor')
        
        # Test environment properties
        assert env.n_player >= 1
        
        print("✅ CliffWalking environment created successfully")
        return env
        
    except ImportError as e:
        pytest.skip(f"Jidi project not available: {e}")
    except Exception as e:
        pytest.fail(f"Failed to create CliffWalking environment: {e}")


def test_cliffwalking_reset():
    """Test environment reset functionality"""
    try:
        config = CliffWalkingConfig(max_steps=50)
        env = CliffWalkingEnv(config)
        
        # Test reset
        obs, info = env.reset(seed=42)
        
        # Verify return types
        assert isinstance(obs, str), f"Expected string observation, got {type(obs)}"
        assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
        
        # Verify observation content
        assert len(obs) > 0, "Observation should not be empty"
        
        # Verify info content
        expected_keys = ["env_name", "step_count", "max_steps", "n_player"]
        for key in expected_keys:
            assert key in info, f"Missing key {key} in info"
        
        assert info["env_name"] == "cliffwalking"
        assert info["step_count"] == 0
        assert info["max_steps"] == 50
        
        print(f"✅ Reset successful")
        print(f"   Observation length: {len(obs)} chars")
        print(f"   Info keys: {list(info.keys())}")
        
        return env, obs, info
        
    except ImportError as e:
        pytest.skip(f"Jidi project not available: {e}")
    except Exception as e:
        pytest.fail(f"Reset test failed: {e}")


def test_cliffwalking_step():
    """Test environment step functionality"""
    try:
        config = CliffWalkingConfig(max_steps=20)
        env = CliffWalkingEnv(config)
        
        # Reset first
        obs, info = env.reset(seed=123)
        
        # Test valid actions
        valid_actions = ["<answer>up</answer>", "<answer>down</answer>", 
                        "<answer>left</answer>", "<answer>right</answer>"]
        
        for i, action in enumerate(valid_actions[:3]):  # Test first 3 actions
            print(f"Testing action {i+1}: {action}")
            
            step_result = env.step(action)
            assert len(step_result) == 5, f"Expected 5-tuple, got {len(step_result)}"
            
            obs, reward, terminated, truncated, info = step_result
            
            # Verify return types
            assert isinstance(obs, str), f"Expected string observation, got {type(obs)}"
            assert isinstance(reward, (int, float)), f"Expected numeric reward, got {type(reward)}"
            assert isinstance(terminated, bool), f"Expected bool terminated, got {type(terminated)}"
            assert isinstance(truncated, bool), f"Expected bool truncated, got {type(truncated)}"
            assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
            
            # Verify info content
            assert "step_count" in info
            assert "action_text" in info
            assert info["step_count"] == i + 1
            assert info["action_text"] == action
            
            print(f"   ✅ Step {i+1} passed - Reward: {reward:.2f}")
            
            if terminated or truncated:
                print(f"   🏁 Episode ended early")
                break
        
        print("✅ Step test successful")
        
    except ImportError as e:
        pytest.skip(f"Jidi project not available: {e}")
    except Exception as e:
        pytest.fail(f"Step test failed: {e}")


def test_cliffwalking_invalid_actions():
    """Test handling of invalid actions"""
    try:
        config = CliffWalkingConfig(max_steps=10, format_penalty=-5.0)
        env = CliffWalkingEnv(config)
        env.reset(seed=456)
        
        # Test invalid actions
        invalid_actions = [
            "invalid action",           # No format
            "<answer></answer>",        # Empty answer
            "<answer>fly</answer>",     # Invalid action
            "just text",               # No tags
            "",                        # Empty string
        ]
        
        for action in invalid_actions:
            print(f"Testing invalid action: '{action}'")
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Invalid actions should receive penalty
            assert reward <= 0, f"Expected penalty for invalid action, got reward: {reward}"
            
            # Check error is recorded
            assert not info.get("action_successful", True) or "error" in info
            
            print(f"   ✅ Correctly penalized with reward: {reward}")
        
        print("✅ Invalid action handling test successful")
        
    except ImportError as e:
        pytest.skip(f"Jidi project not available: {e}")
    except Exception as e:
        pytest.fail(f"Invalid action test failed: {e}")


def test_cliffwalking_methods():
    """Test other environment methods"""
    try:
        config = CliffWalkingConfig()
        env = CliffWalkingEnv(config)
        
        # Test get_all_actions
        actions = env.get_all_actions()
        assert isinstance(actions, list), f"Expected list, got {type(actions)}"
        assert len(actions) > 0, "Should have at least one action"
        
        expected_actions = ["up", "down", "left", "right"]
        for action in expected_actions:
            assert action in actions, f"Missing expected action: {action}"
        
        print(f"✅ Available actions: {actions}")
        
        # Test render
        env.reset()
        render_result = env.render(mode="text")
        assert isinstance(render_result, str), f"Expected string render, got {type(render_result)}"
        assert len(render_result) > 0, "Render result should not be empty"
        
        print("✅ Render test successful")
        
        # Test parse_action
        parsed = env.parse_action("<answer>up</answer>")
        # Should not throw exception (result depends on implementation)
        
        print("✅ Parse action test successful")
        
        # Test stats
        stats = env.get_stats()
        assert isinstance(stats, dict), f"Expected dict stats, got {type(stats)}"
        assert "total_steps" in stats
        assert "success_rate" in stats
        
        print(f"✅ Stats: {stats}")
        
        # Test close
        env.close()  # Should not throw exception
        print("✅ Close test successful")
        
    except ImportError as e:
        pytest.skip(f"Jidi project not available: {e}")
    except Exception as e:
        pytest.fail(f"Methods test failed: {e}")


def test_cliffwalking_episode():
    """Test a complete episode"""
    try:
        config = CliffWalkingConfig(max_steps=30)
        env = CliffWalkingEnv(config)
        
        obs, info = env.reset(seed=789)
        
        # Run a short episode with simple strategy
        actions = ["<answer>right</answer>"] * 5 + ["<answer>up</answer>"] * 2
        
        total_reward = 0.0
        step_count = 0
        
        for action in actions:
            step_count += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step_count}: action={action}, reward={reward:.2f}, done={terminated or truncated}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step_count}")
                break
        
        print(f"✅ Episode test successful")
        print(f"   Total steps: {step_count}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward: {total_reward/step_count:.3f}")
        
        # Get final stats
        final_stats = env.get_stats()
        print(f"   Final stats: {final_stats}")
        
        env.close()
        
    except ImportError as e:
        pytest.skip(f"Jidi project not available: {e}")
    except Exception as e:
        pytest.fail(f"Episode test failed: {e}")


def main():
    """Run all tests manually (for debugging)"""
    print("🧪 Testing Jidi CliffWalking integration with ROLL...")
    print("=" * 60)
    
    tests = [
        ("Registration", test_jidi_cliffwalking_registration),
        ("Configuration", test_cliffwalking_config),
        ("Environment Creation", test_cliffwalking_env_creation),
        ("Reset", test_cliffwalking_reset),
        ("Step", test_cliffwalking_step),
        ("Invalid Actions", test_cliffwalking_invalid_actions),
        ("Methods", test_cliffwalking_methods),
        ("Episode", test_cliffwalking_episode),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests PASSED! Jidi CliffWalking is ready for ROLL training!")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
