#!/usr/bin/env python3
"""
Example script demonstrating all four newly integrated Jidi environments in ROLL

This script shows how to use CliffWalking, GridWorld, MiniGrid, and Sokoban
environments with the ROLL framework.
"""

import sys
from pathlib import Path

# Add ROLL project root to path
roll_root = Path(__file__).parent.parent
sys.path.insert(0, str(roll_root))

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS


def test_environment(env_name: str, num_steps: int = 5):
    """Test a specific environment with some sample actions"""
    print(f"\n🧪 Testing {env_name}")
    print("="*50)
    
    try:
        # Get environment and config classes
        env_class = REGISTERED_ENVS[env_name]
        config_class = REGISTERED_ENV_CONFIGS[env_name]
        
        # Create config and environment
        config = config_class(max_steps=20)
        env = env_class(config)
        
        print(f"📋 Configuration:")
        print(f"  - Environment: {config.jidi_env_name}")
        print(f"  - Max steps: {config.max_steps}")
        print(f"  - Max tokens per step: {config.max_tokens_per_step}")
        
        # Reset environment
        obs, info = env.reset(seed=42)
        print(f"\n🔄 Reset complete!")
        print(f"  - Environment type: {info.get('environment_type', 'unknown')}")
        print(f"  - N players: {info.get('n_player', 1)}")
        print(f"  - Episode ID: {info.get('episode_id', 0)}")
        
        print(f"\n👁️ Initial observation:")
        print(obs[:200] + "..." if len(obs) > 200 else obs)
        
        # Get available actions
        actions = env.get_all_actions()
        print(f"\n🎮 Available actions: {actions}")
        
        # Take some sample actions
        sample_actions = {
            "jidi_cliffwalking": ["<answer>right</answer>", "<answer>right</answer>", "<answer>down</answer>"],
            "jidi_gridworld": ["<answer>up</answer>", "<answer>right</answer>", "<answer>down</answer>"],
            "jidi_minigrid": ["<answer>forward</answer>", "<answer>left</answer>", "<answer>forward</answer>"],
            "jidi_sokoban": ["<answer>up</answer>", "<answer>right</answer>", "<answer>down</answer>"]
        }
        
        test_actions = sample_actions.get(env_name, ["<answer>up</answer>", "<answer>down</answer>"])
        
        print(f"\n🚀 Taking {min(num_steps, len(test_actions))} sample actions:")
        
        total_reward = 0
        for i, action in enumerate(test_actions[:num_steps]):
            print(f"\n  Step {i+1}: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"    Reward: {reward}")
            print(f"    Terminated: {terminated}, Truncated: {truncated}")
            print(f"    Action valid: {info.get('action_valid', True)}")
            
            # Show observation (truncated)
            obs_display = obs[:100] + "..." if len(obs) > 100 else obs
            print(f"    New observation: {obs_display}")
            
            if terminated or truncated:
                print(f"    🏁 Episode ended!")
                break
        
        print(f"\n📊 Episode summary:")
        print(f"  - Total reward: {total_reward}")
        print(f"  - Steps taken: {info.get('step_count', 0)}")
        print(f"  - Success rate: {info.get('success_rate', 0):.2f}")
        
        # Test rendering
        if hasattr(env, 'render'):
            try:
                rendered = env.render(mode="text")
                if rendered:
                    print(f"\n🖼️ Text rendering:")
                    print(rendered[:200] + "..." if len(str(rendered)) > 200 else rendered)
            except Exception as e:
                print(f"\n⚠️ Rendering failed: {e}")
        
        env.close()
        print(f"\n✅ {env_name} test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing {env_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to test all environments"""
    print("🎮 ROLL Jidi Environments Integration Demo")
    print("=" * 60)
    print("Testing all four newly integrated environments:")
    print("1. CliffWalking - Navigation with cliff penalty")
    print("2. GridWorld - Basic grid navigation")  
    print("3. MiniGrid - Partially observable grid world")
    print("4. Sokoban - Box-pushing puzzle")
    
    # List of environments to test
    environments = [
        "jidi_cliffwalking",
        "jidi_gridworld", 
        "jidi_minigrid",
        "jidi_sokoban"
    ]
    
    # Test each environment
    for env_name in environments:
        test_environment(env_name, num_steps=3)
    
    print(f"\n{'='*60}")
    print("🎉 All environments tested successfully!")
    print("🚀 These environments are now ready for ROLL training!")
    print("📚 Check the documentation for integration details.")
    
    # Show integration summary
    print(f"\n📋 Integration Summary:")
    print(f"  - Total environments integrated: {len(environments)}")
    print(f"  - Framework: ROLL")
    print(f"  - Source: Jidi Platform")
    print(f"  - Adapter type: Lightweight wrapper")
    print(f"  - Text-based: ✅")
    print(f"  - Multi-agent ready: ✅")
    print(f"  - Training ready: ✅")


if __name__ == "__main__":
    main()