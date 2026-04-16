#!/usr/bin/env python3
"""
Comprehensive test script for all Jidi environments in ROLL framework

Tests the integration of CliffWalking, GridWorld, MiniGrid, and Sokoban
environments with ROLL's training pipeline.
"""

import pytest
import sys
import os
from pathlib import Path

# Add ROLL project root to path
roll_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(roll_root))

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS


# Test data for all environments
JIDI_ENVIRONMENTS = {
    "jidi_cliffwalking": {
        "config_class": "CliffWalkingConfig",
        "env_class": "CliffWalkingEnv", 
        "test_actions": ["<answer>up</answer>", "<answer>right</answer>", "<answer>down</answer>"],
        "expected_actions": ["up", "down", "left", "right"],
        "environment_type": "navigation"
    },
    "jidi_gridworld": {
        "config_class": "GridWorldConfig",
        "env_class": "GridWorldEnv",
        "test_actions": ["<answer>up</answer>", "<answer>right</answer>", "<answer>left</answer>"],
        "expected_actions": ["up", "down", "left", "right"],
        "environment_type": "navigation"
    },
    "jidi_minigrid": {
        "config_class": "MiniGridConfig", 
        "env_class": "MiniGridEnv",
        "test_actions": ["<answer>forward</answer>", "<answer>left</answer>", "<answer>right</answer>"],
        "expected_actions": ["forward", "left", "right", "pickup", "drop", "toggle", "done"],
        "environment_type": "minigrid"
    },
    "jidi_sokoban": {
        "config_class": "SokobanConfig",
        "env_class": "SokobanEnv",
        "test_actions": ["<answer>up</answer>", "<answer>down</answer>", "<answer>right</answer>"],
        "expected_actions": ["up", "down", "left", "right"],
        "environment_type": "puzzle"
    }
}


class TestJidiEnvironmentRegistration:
    """Test environment registration in ROLL"""
    
    def test_all_environments_registered(self):
        """Test that all Jidi environments are registered"""
        for env_name in JIDI_ENVIRONMENTS.keys():
            assert env_name in REGISTERED_ENVS, f"Environment {env_name} not registered"
            assert env_name in REGISTERED_ENV_CONFIGS, f"Config {env_name} not registered"
    
    def test_environment_classes(self):
        """Test that environment classes are correct"""
        for env_name, env_data in JIDI_ENVIRONMENTS.items():
            env_class = REGISTERED_ENVS[env_name]
            config_class = REGISTERED_ENV_CONFIGS[env_name]
            
            assert env_class.__name__ == env_data["env_class"]
            assert config_class.__name__ == env_data["config_class"]


class TestJidiEnvironmentConfigs:
    """Test environment configurations"""
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_config_creation(self, env_name, env_data):
        """Test configuration creation for each environment"""
        config_class = REGISTERED_ENV_CONFIGS[env_name]
        config = config_class()
        
        # Test basic properties
        assert hasattr(config, 'jidi_env_name')
        assert hasattr(config, 'max_steps')
        assert hasattr(config, 'max_tokens_per_step')
        assert hasattr(config, 'env_instruction')
        
        # Test instruction is not empty
        assert len(config.env_instruction) > 0
        
        # Test reasonable defaults
        assert config.max_steps > 0
        assert config.max_tokens_per_step > 0
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_config_customization(self, env_name, env_data):
        """Test configuration customization"""
        config_class = REGISTERED_ENV_CONFIGS[env_name]
        
        custom_config = config_class(
            max_steps=150,
            max_tokens_per_step=64,
            format_penalty=-3.0
        )
        
        assert custom_config.max_steps == 150
        assert custom_config.max_tokens_per_step == 64
        assert custom_config.format_penalty == -3.0


class TestJidiEnvironmentCreation:
    """Test environment creation"""
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_environment_creation(self, env_name, env_data):
        """Test creating each environment"""
        try:
            config_class = REGISTERED_ENV_CONFIGS[env_name]
            env_class = REGISTERED_ENVS[env_name]
            
            config = config_class(max_steps=50)
            env = env_class(config)
            
            assert env is not None
            assert hasattr(env, 'jidi_env')
            assert hasattr(env, 'interactor')
            assert env.n_player >= 1
            
            env.close()
            
        except ImportError:
            pytest.skip(f"Jidi project not available for {env_name}")
        except Exception as e:
            pytest.fail(f"Failed to create {env_name}: {e}")


class TestJidiEnvironmentFunctionality:
    """Test environment functionality"""
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_reset_functionality(self, env_name, env_data):
        """Test reset functionality for each environment"""
        try:
            config_class = REGISTERED_ENV_CONFIGS[env_name]
            env_class = REGISTERED_ENVS[env_name]
            
            config = config_class(max_steps=50)
            env = env_class(config)
            
            # Test reset
            obs, info = env.reset(seed=42)
            
            # Verify return types
            assert isinstance(obs, str)
            assert isinstance(info, dict)
            
            # Verify required info fields
            required_fields = ["env_name", "step_count", "max_steps", "n_player", "environment_type"]
            for field in required_fields:
                assert field in info, f"Missing field {field} in {env_name}"
            
            # Verify environment-specific info
            assert info["environment_type"] == env_data["environment_type"]
            assert info["step_count"] == 0
            assert info["max_steps"] == 50
            
            env.close()
            
        except ImportError:
            pytest.skip(f"Jidi project not available for {env_name}")
        except Exception as e:
            pytest.fail(f"Reset test failed for {env_name}: {e}")
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_step_functionality(self, env_name, env_data):
        """Test step functionality for each environment"""
        try:
            config_class = REGISTERED_ENV_CONFIGS[env_name]
            env_class = REGISTERED_ENVS[env_name]
            
            config = config_class(max_steps=20)
            env = env_class(config)
            
            # Reset first
            obs, info = env.reset(seed=123)
            
            # Test valid actions
            for i, action in enumerate(env_data["test_actions"][:3]):
                step_result = env.step(action)
                assert len(step_result) == 5
                
                obs, reward, terminated, truncated, info = step_result
                
                # Verify return types
                assert isinstance(obs, str)
                assert isinstance(reward, (int, float))
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)
                
                # Verify step info (step_count might be different if episode ended early)
                assert info["step_count"] >= i + 1
                assert "action_text" in info
                
                if terminated or truncated:
                    break
            
            env.close()
            
        except ImportError:
            pytest.skip(f"Jidi project not available for {env_name}")
        except Exception as e:
            pytest.fail(f"Step test failed for {env_name}: {e}")
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_get_all_actions(self, env_name, env_data):
        """Test get_all_actions for each environment"""
        try:
            config_class = REGISTERED_ENV_CONFIGS[env_name]
            env_class = REGISTERED_ENVS[env_name]
            
            config = config_class()
            env = env_class(config)
            
            actions = env.get_all_actions()
            assert isinstance(actions, list)
            assert len(actions) > 0
            
            # Check expected actions are present
            for expected_action in env_data["expected_actions"]:
                if expected_action not in ["coordinate_input", "chess_notation"]:  # Skip special markers
                    assert expected_action in actions, f"Missing action {expected_action} in {env_name}"
            
            env.close()
            
        except ImportError:
            pytest.skip(f"Jidi project not available for {env_name}")
        except Exception as e:
            pytest.fail(f"Actions test failed for {env_name}: {e}")


class TestJidiEnvironmentErrorHandling:
    """Test error handling"""
    
    @pytest.mark.parametrize("env_name,env_data", JIDI_ENVIRONMENTS.items())
    def test_invalid_actions(self, env_name, env_data):
        """Test invalid action handling"""
        try:
            config_class = REGISTERED_ENV_CONFIGS[env_name]
            env_class = REGISTERED_ENVS[env_name]
            
            config = config_class(max_steps=10, format_penalty=-2.0)
            env = env_class(config)
            env.reset(seed=456)
            
            invalid_actions = [
                "invalid action",
                "<answer></answer>",
                "<answer>fly</answer>",
                "just text",
                "",
            ]
            
            for action in invalid_actions:
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Invalid actions should typically receive penalty
                # (Some environments might handle gracefully)
                assert isinstance(reward, (int, float))
                assert isinstance(info, dict)
                
                # Check that we get some kind of response
                assert len(obs) > 0
            
            env.close()
            
        except ImportError:
            pytest.skip(f"Jidi project not available for {env_name}")
        except Exception as e:
            pytest.fail(f"Error handling test failed for {env_name}: {e}")


def run_comprehensive_test():
    """Run comprehensive test suite for all Jidi environments"""
    print("🧪 Running Comprehensive Jidi Environment Tests")
    print("=" * 60)
    
    test_classes = [
        TestJidiEnvironmentRegistration(),
        TestJidiEnvironmentConfigs(),
        TestJidiEnvironmentCreation(),
        TestJidiEnvironmentFunctionality(),
        TestJidiEnvironmentErrorHandling(),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n🔍 Running {class_name}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            method = getattr(test_class, method_name)
            
            if hasattr(method, 'pytestmark'):  # Parametrized test
                # Run for each environment
                for env_name, env_data in JIDI_ENVIRONMENTS.items():
                    total_tests += 1
                    try:
                        method(env_name, env_data)
                        print(f"  ✅ {method_name}[{env_name}]")
                        passed_tests += 1
                    except Exception as e:
                        print(f"  ❌ {method_name}[{env_name}]: {e}")
            else:
                # Regular test method
                total_tests += 1
                try:
                    method()
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests PASSED! All Jidi environments are ready for ROLL training!")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
