#!/usr/bin/env python3
"""
Example: Using Jidi CliffWalking environment in ROLL framework

This example demonstrates how to use the integrated Jidi CliffWalking environment
within the ROLL training framework.
"""

import sys
import os
from pathlib import Path

# Add ROLL to path
roll_root = Path(__file__).parent.parent
sys.path.insert(0, str(roll_root))

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS


def basic_usage_example():
    """Basic usage example of Jidi CliffWalking in ROLL"""
    print("🎮 Jidi CliffWalking in ROLL - Basic Usage Example")
    print("=" * 60)
    
    try:
        # Get registered environment class and config
        env_class = REGISTERED_ENVS["jidi_cliffwalking"]
        config_class = REGISTERED_ENV_CONFIGS["jidi_cliffwalking"]
        
        print(f"✅ Found environment class: {env_class}")
        print(f"✅ Found config class: {config_class}")
        
        # Create configuration
        config = config_class(
            max_steps=100,
            max_tokens_per_step=64,
            format_penalty=-2.0
        )
        print(f"✅ Created config: {config.jidi_env_name}")
        
        # Create environment
        env = env_class(config)
        print(f"✅ Created environment with {env.n_player} player(s)")
        
        # Reset environment
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset successful")
        print(f"   Initial observation: {obs[:100]}...")
        print(f"   Info: {info}")
        
        # Run a few steps
        actions = ["<answer>right</answer>", "<answer>up</answer>", "<answer>right</answer>"]
        
        total_reward = 0.0
        for i, action in enumerate(actions):
            print(f"\n🎯 Step {i+1}: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Reward: {reward:.2f}")
            print(f"   Terminated: {terminated}, Truncated: {truncated}")
            print(f"   Observation: {obs[:100]}...")
            
            if terminated or truncated:
                print("   🏁 Episode finished!")
                break
        
        print(f"\n📊 Episode Summary:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Steps taken: {i+1}")
        
        # Get environment statistics
        stats = env.get_stats()
        print(f"   Environment stats: {stats}")
        
        # Available actions
        actions = env.get_all_actions()
        print(f"   Available actions: {actions}")
        
        # Close environment
        env.close()
        print("✅ Environment closed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error (Jidi project not available): {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def config_customization_example():
    """Example showing configuration customization"""
    print("\n🔧 Configuration Customization Example")
    print("=" * 60)
    
    try:
        config_class = REGISTERED_ENV_CONFIGS["jidi_cliffwalking"]
        
        # Create custom configuration
        custom_config = config_class(
            jidi_env_name="cliffwalking",
            max_steps=50,
            max_tokens_per_step=32,
            format_penalty=-5.0,
            invalid_action_penalty=-1.0,
            env_instruction=(
                "You are a brave explorer navigating a treacherous cliff path. "
                "Your mission is to reach the treasure at the end while avoiding "
                "the deadly cliff. Each step counts!"
            ),
            cliff_penalty=-50.0,  # Custom cliff penalty
            step_penalty=-0.5,    # Custom step penalty
        )
        
        print(f"✅ Custom configuration created")
        print(f"   Environment: {custom_config.jidi_env_name}")
        print(f"   Max steps: {custom_config.max_steps}")
        print(f"   Cliff penalty: {custom_config.cliff_penalty}")
        print(f"   Custom instruction: {custom_config.env_instruction[:100]}...")
        
        # Create environment with custom config
        env_class = REGISTERED_ENVS["jidi_cliffwalking"]
        env = env_class(custom_config)
        
        # Test with custom config
        obs, info = env.reset(seed=123)
        print(f"✅ Custom environment working")
        print(f"   Grid size: {info.get('grid_size', 'N/A')}")
        print(f"   Cliff penalty: {info.get('cliff_penalty', 'N/A')}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def integration_check():
    """Check ROLL integration status"""
    print("\n🔌 ROLL Integration Status")
    print("=" * 60)
    
    print("Registered Environments:")
    for env_name, env_class in REGISTERED_ENVS.items():
        if "jidi" in env_name.lower():
            print(f"  ✅ {env_name}: {env_class}")
    
    print("\nRegistered Configurations:")
    for config_name, config_class in REGISTERED_ENV_CONFIGS.items():
        if "jidi" in config_name.lower():
            print(f"  ✅ {config_name}: {config_class}")
    
    # Check if ready for training
    jidi_envs = [name for name in REGISTERED_ENVS.keys() if "jidi" in name.lower()]
    jidi_configs = [name for name in REGISTERED_ENV_CONFIGS.keys() if "jidi" in name.lower()]
    
    if jidi_envs and jidi_configs:
        print(f"\n🎉 ROLL integration successful!")
        print(f"   {len(jidi_envs)} Jidi environment(s) available for training")
        print(f"   Ready to use with ROLL's agentic training pipeline")
        return True
    else:
        print(f"\n⚠️ Integration incomplete")
        return False


def main():
    """Main example function"""
    print("🚀 Jidi Environment Integration with ROLL Framework")
    print("=" * 70)
    
    # Run examples
    success1 = basic_usage_example()
    success2 = config_customization_example()
    success3 = integration_check()
    
    if success1 and success2 and success3:
        print(f"\n✅ All examples completed successfully!")
        print(f"📚 Next steps:")
        print(f"   1. Use 'jidi_cliffwalking' in ROLL training configurations")
        print(f"   2. Customize configs in custom_envs section")
        print(f"   3. Run ROLL agentic training pipeline")
        print(f"   4. Add more Jidi environments as needed")
        return 0
    else:
        print(f"\n❌ Some examples failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
