"""
Jidi Environment Integration for ROLL Framework

This module provides the base environment class and specific implementations
for integrating Jidi environments with ROLL's training pipeline.
"""

import sys
import os
import traceback
from typing import Any, Dict, List, Tuple, Optional, Union

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.parse_action_utils import default_parser_action_func
from .config import JidiBaseConfig, CliffWalkingConfig, GridWorldConfig, MiniGridConfig, SokobanConfig


class JidiBaseEnv(BaseEnv):
    """
    Base class for all Jidi environments in ROLL
    
    This class handles the integration between Jidi's environment system
    and ROLL's training framework by:
    1. Loading Jidi environments and interactors
    2. Converting between text and environment formats
    3. Handling errors and edge cases
    4. Providing consistent ROLL-compatible interface
    """
    
    def __init__(self, config: JidiBaseConfig):
        super().__init__(config)
        self.config: JidiBaseConfig = config
        
        # Add Jidi project to Python path
        if config.jidi_project_path not in sys.path:
            sys.path.insert(0, config.jidi_project_path)
        
        # Import Jidi modules
        try:
            from ai_lib.env.chooseenv import make
            from ai_lib.interactors import LLM_INTERACTOR_REGISTRY
            self._make_env = make
            self._interactor_registry = LLM_INTERACTOR_REGISTRY
        except ImportError as e:
            raise ImportError(f"Failed to import Jidi modules. Please check jidi_project_path: {e}")
        
        # Create Jidi environment and interactor
        self._initialize_jidi_components()
        
        # State tracking
        self._last_obs = None
        self._last_text_obs = ""
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_done = False
        
        # Statistics
        self._stats = {
            "total_steps": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "format_errors": 0,
            "episodes": 0
        }
    
    def _initialize_jidi_components(self):
        """Initialize Jidi environment and interactor"""
        try:
            if self.config.textualize:
                # When textualize=True, make() returns an interactor directly
                self.interactor = self._make_env(
                    self.config.jidi_env_name, 
                    textualize=True
                )
                # Get the underlying environment from the interactor
                self.jidi_env = self.interactor.game
            else:
                # When textualize=False, make() returns the raw environment
                self.jidi_env = self._make_env(
                    self.config.jidi_env_name, 
                    textualize=False
                )
                
                # Create LLM interactor manually
                if self.config.jidi_env_name not in self._interactor_registry:
                    raise ValueError(f"Environment {self.config.jidi_env_name} not found in interactor registry")
                
                interactor_class = self._interactor_registry[self.config.jidi_env_name]
                self.interactor = interactor_class(self.jidi_env)
            
            # Store environment info
            self.n_player = getattr(self.interactor, 'n_player', 1)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Jidi components: {e}")
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """
        Reset the environment
        
        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
            
        Returns:
            Tuple[str, dict]: (text_observation, info_dict)
        """
        try:
            # Reset counters
            self._step_count = 0
            self._episode_reward = 0.0
            self._episode_done = False
            self._stats["episodes"] += 1
            
            # Reset Jidi environment through interactor
            text_obs = self.interactor.reset()
            
            # Handle single/multi-agent format
            if isinstance(text_obs, list):
                self._last_text_obs = text_obs[0] if text_obs else ""
            else:
                self._last_text_obs = text_obs if text_obs else ""
            
            # Create info dict
            info = {
                "env_name": self.config.jidi_env_name,
                "step_count": self._step_count,
                "max_steps": self.config.max_steps,
                "n_player": self.n_player,
                "episode_id": self._stats["episodes"],
                "metrics": {
                    "action_is_valid": True,
                    "action_is_effective": True,
                    "success": False  # Reset state, no success yet
                }
            }
            
            return self._last_text_obs, info
            
        except Exception as e:
            error_msg = f"Reset failed: {str(e)}"
            error_info = {
                "error": True, 
                "error_message": error_msg,
                "env_name": self.config.jidi_env_name,
                "step_count": 0,
                "max_steps": self.config.max_steps,
                "n_player": getattr(self, 'n_player', 1),
                "episode_id": self._stats.get("episodes", 0),
                "metrics": {
                    "action_is_valid": False,
                    "action_is_effective": False,
                    "success": False
                }
            }
            return f"Error: {error_msg}", error_info
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Text action from LLM
            
        Returns:
            Tuple[str, float, bool, bool, Dict]: (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        self._stats["total_steps"] += 1
        
        # Initialize return values
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "step_count": self._step_count,
            "action_text": action,
            "action_valid": True,
            "action_successful": True,
            "error_message": None
        }
        
        try:
            # Parse action using simple regex extraction since ROLL's parser 
            # expects string-to-string mapping but Jidi uses string-to-int
            import re
            
            # First try to extract from answer tags
            match = re.search(r'<answer>(.*?)</answer>', action, re.IGNORECASE)
            if match:
                clean_action = match.group(1).strip()
            else:
                # Use the action as-is if no tags found
                clean_action = action.strip()
            
            # Validate the clean action against available actions
            available_actions = self.get_all_actions()
            if clean_action not in available_actions:
                # Log debug info for troubleshooting
                self.logger.debug(f"Action '{clean_action}' not in available actions: {available_actions}")
                self.logger.debug(f"Original action: '{action}'")
                # Still try the action, let Jidi handle the error
            
            # Execute action through interactor with the clean action
            step_result = self.interactor.step(clean_action)
            
            # Parse result (should be 5-tuple)
            if len(step_result) == 5:
                text_obs, reward, terminated, truncated, step_info = step_result
            else:
                raise ValueError(f"Unexpected step result format: {len(step_result)} elements")
            
            # Handle single/multi-agent format
            if isinstance(text_obs, list):
                self._last_text_obs = text_obs[0] if text_obs else ""
            else:
                self._last_text_obs = text_obs if text_obs else ""
            
            # Handle reward format
            if isinstance(reward, list):
                reward = reward[0] if reward else 0.0
            
            # Update episode tracking
            self._episode_reward += reward
            
            # Merge step info
            if isinstance(step_info, dict):
                info.update(step_info)
            
            # Check max steps
            if self._step_count >= self.config.max_steps:
                truncated = True
            
            self._stats["successful_actions"] += 1
            
        except Exception as e:
            # Handle action execution errors
            error_msg = str(e)
            self._stats["failed_actions"] += 1
            
            # Determine penalty type
            if any(keyword in error_msg.lower() for keyword in ["parsing", "format", "invalid action"]):
                reward = self.config.format_penalty
                self._stats["format_errors"] += 1
                info["format_error"] = True
            else:
                reward = self.config.invalid_action_penalty
                info["action_valid"] = False
            
            info.update({
                "action_successful": False,
                "error_message": error_msg,
                "penalty_applied": reward
            })
            
            # Check max steps
            if self._step_count >= self.config.max_steps:
                truncated = True
        
        # Mark episode as done
        if terminated or truncated:
            self._episode_done = True
        
        # Add required metrics field for ROLL framework
        metrics = {
            "action_is_valid": info.get("action_valid", True),
            "action_is_effective": info.get("action_successful", True),
            "success": terminated and reward > 0,  # Success if episode ended with positive reward
        }
        info["metrics"] = metrics
        
        return self._last_text_obs, float(reward), terminated, truncated, info
    
    def parse_action(self, text: str) -> Any:
        """
        Parse text action using default parser and action pattern
        
        Args:
            text: Raw text response from LLM
            
        Returns:
            Parsed action or None if parsing fails
        """
        try:
            # Use ROLL's default parser with Jidi-specific pattern
            action_info = default_parser_action_func(
                text, 
                self.config.action_pattern,
                getattr(self.interactor, 'action_map', {}),
                self.config.special_token_list
            )
            return action_info.get("action")
        except Exception as e:
            return None
    
    def render(self, mode: str = "text") -> Any:
        """
        Render the environment
        
        Args:
            mode: Render mode ("text", "rgb_array", etc.)
            
        Returns:
            Rendered output
        """
        if mode == "text":
            return self._last_text_obs
        else:
            # Try to render through original environment
            try:
                if hasattr(self.jidi_env, 'render'):
                    return self.jidi_env.render()
                else:
                    return self._last_text_obs
            except Exception as e:
                return f"Render error: {e}"
    
    def get_all_actions(self) -> List[str]:
        """
        Get all valid actions for this environment
        
        Returns:
            List of valid action strings
        """
        try:
            # Try to get actions from interactor
            if hasattr(self.interactor, 'action_map'):
                return list(self.interactor.action_map.keys())
            elif hasattr(self.interactor, 'get_all_actions'):
                return self.interactor.get_all_actions()
            else:
                # Fallback to environment-specific defaults
                return self._get_default_actions()
        except Exception:
            return self._get_default_actions()
    
    def _get_default_actions(self) -> List[str]:
        """Get default actions based on environment type"""
        action_maps = {
            "cliffwalking": ["up", "down", "left", "right"],
            "gridworld": ["up", "down", "left", "right"],
            "sokoban_1p": ["up", "down", "left", "right"],
            "snakes_1v1": ["up", "down", "left", "right"],
            "seek_2p": ["up", "down", "left", "right"],
            "gobang_1v1": ["coordinate_input"],
            "reversi_1v1": ["coordinate_input"],
            "chinesechess": ["chess_notation"],
            "delivery_two_agents": ["move up", "move down", "move left", "move right", "grab orders", "deliver orders"],
            # MiniGrid environments
            "MiniGrid-DoorKey-16x16-v0": ["forward", "left", "right", "pickup", "drop", "toggle", "done"],
            "MiniGrid-MultiRoom-N6-v0": ["forward", "left", "right", "pickup", "drop", "toggle", "done"],
            "MiniGrid-Dynamic-Obstacles-16x16-v0": ["forward", "left", "right", "pickup", "drop", "toggle", "done"],
        }
        return action_maps.get(self.config.jidi_env_name, ["action"])
    
    def close(self):
        """Close the environment and cleanup"""
        try:
            if hasattr(self.jidi_env, 'close'):
                self.jidi_env.close()
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics"""
        if self._stats["total_steps"] > 0:
            success_rate = self._stats["successful_actions"] / self._stats["total_steps"]
        else:
            success_rate = 0.0
            
        return {
            **self._stats,
            "episode_reward": self._episode_reward,
            "step_count": self._step_count,
            "success_rate": success_rate,
            "episode_done": self._episode_done
        }


class CliffWalkingEnv(JidiBaseEnv):
    """
    CliffWalking environment for ROLL
    
    A grid-world navigation task where the agent must reach a goal
    while avoiding a dangerous cliff that gives large negative rewards.
    """
    
    def __init__(self, config: CliffWalkingConfig = None):
        if config is None:
            config = CliffWalkingConfig()
        super().__init__(config)
        self.config: CliffWalkingConfig = config
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """Reset with CliffWalking-specific information"""
        text_obs, info = super().reset(seed=seed, **kwargs)
        
        # Add CliffWalking-specific info
        info.update({
            "environment_type": "navigation",
            "grid_size": f"{self.config.grid_height}x{self.config.grid_width}",
            "cliff_penalty": self.config.cliff_penalty,
            "step_penalty": self.config.step_penalty,
            "goal_reward": self.config.goal_reward
        })
        
        return text_obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """Step with CliffWalking-specific handling"""
        text_obs, reward, terminated, truncated, info = super().step(action)
        
        # Add CliffWalking-specific info
        info.update({
            "cliff_hit": reward == self.config.cliff_penalty,
            "goal_reached": terminated and reward >= 0,
            "cumulative_reward": self._episode_reward
        })
        
        return text_obs, reward, terminated, truncated, info


class GridWorldEnv(JidiBaseEnv):
    """
    GridWorld environment for ROLL
    
    A grid-world navigation task where the agent must navigate
    through a grid to reach a goal position.
    """
    
    def __init__(self, config: GridWorldConfig = None):
        if config is None:
            config = GridWorldConfig()
        super().__init__(config)
        self.config: GridWorldConfig = config
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """Reset with GridWorld-specific information"""
        text_obs, info = super().reset(seed=seed, **kwargs)
        
        # Add GridWorld-specific info
        info.update({
            "environment_type": "navigation",
            "grid_info": self.config.grid_size,
            "step_penalty": self.config.step_penalty,
            "goal_reward": self.config.goal_reward
        })
        
        return text_obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """Step with GridWorld-specific handling"""
        text_obs, reward, terminated, truncated, info = super().step(action)
        
        # Add GridWorld-specific info
        info.update({
            "goal_reached": terminated and reward > 0,
            "cumulative_reward": self._episode_reward
        })
        
        return text_obs, reward, terminated, truncated, info


class MiniGridEnv(JidiBaseEnv):
    """
    MiniGrid environment for ROLL
    
    A partially observable grid-world environment with various
    objectives like navigation, picking up objects, opening doors.
    """
    
    def __init__(self, config: MiniGridConfig = None):
        if config is None:
            config = MiniGridConfig()
        super().__init__(config)
        self.config: MiniGridConfig = config
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """Reset with MiniGrid-specific information"""
        text_obs, info = super().reset(seed=seed, **kwargs)
        
        # Add MiniGrid-specific info
        info.update({
            "environment_type": "minigrid",
            "grid_size": self.config.grid_size,
            "render_mode": self.config.render_mode,
            "partial_observability": True
        })
        
        return text_obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """Step with MiniGrid-specific handling"""
        text_obs, reward, terminated, truncated, info = super().step(action)
        
        # Add MiniGrid-specific info
        info.update({
            "objective_completed": terminated and reward > 0,
            "cumulative_reward": self._episode_reward
        })
        
        return text_obs, reward, terminated, truncated, info
    
    def _get_default_actions(self) -> List[str]:
        """MiniGrid specific actions"""
        return ["forward", "left", "right", "pickup", "drop", "toggle", "done"]


class SokobanEnv(JidiBaseEnv):
    """
    Sokoban environment for ROLL
    
    A puzzle game where the player must push boxes to target
    positions. Requires strategic planning to avoid getting stuck.
    """
    
    def __init__(self, config: SokobanConfig = None):
        if config is None:
            config = SokobanConfig()
        super().__init__(config)
        self.config: SokobanConfig = config
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """Reset with Sokoban-specific information"""
        text_obs, info = super().reset(seed=seed, **kwargs)
        
        # Add Sokoban-specific info
        info.update({
            "environment_type": "puzzle",
            "difficulty": self.config.puzzle_difficulty,
            "box_penalty": self.config.box_penalty,
            "target_reward": self.config.target_reward
        })
        
        return text_obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """Step with Sokoban-specific handling"""
        text_obs, reward, terminated, truncated, info = super().step(action)
        
        # Add Sokoban-specific info
        info.update({
            "puzzle_solved": terminated and reward > 0,
            "boxes_on_targets": info.get("boxes_on_target", 0),
            "cumulative_reward": self._episode_reward
        })
        
        return text_obs, reward, terminated, truncated, info


# Environment registry for easy lookup
JIDI_ENV_REGISTRY = {
    "jidi_cliffwalking": CliffWalkingEnv,
    "jidi_gridworld": GridWorldEnv,
    "jidi_minigrid": MiniGridEnv,
    "jidi_sokoban": SokobanEnv,
    # Add more environments as they are implemented
}
