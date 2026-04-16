"""
Configuration classes for Jidi environments in ROLL framework
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from roll.agentic.env.base import BaseEnvConfig


@dataclass
class JidiBaseConfig(BaseEnvConfig):
    """
    Base configuration for all Jidi environments in ROLL
    
    This class provides common configuration options for Jidi environments
    while maintaining compatibility with ROLL's BaseEnvConfig.
    """
    
    # Jidi specific configurations
    jidi_env_name: str = field(default="cliffwalking", metadata={"help": "Original Jidi environment name"})
    textualize: bool = field(default=True, metadata={"help": "Enable text-based interaction"})
    
    # ROLL standard configurations  
    max_steps: int = field(default=100, metadata={"help": "Maximum steps per episode"})
    env_instruction: str = field(default="", metadata={"help": "Environment instruction for the agent"})
    action_pattern: str = field(default=r"<answer>(.*?)</answer>", metadata={"help": "Regex pattern for action parsing"})
    max_tokens_per_step: int = field(default=128, metadata={"help": "Maximum tokens per step"})
    
    # Error handling
    format_penalty: float = field(default=-1.0, metadata={"help": "Penalty for incorrect action format"})
    invalid_action_penalty: float = field(default=-0.1, metadata={"help": "Penalty for invalid actions"})
    
    # Jidi environment path
    jidi_project_path: str = field(
        default="/data1/Weiyu_project/ai_lib_dev", 
        metadata={"help": "Path to Jidi project root"}
    )
    
    # Special tokens for parsing
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>"],
        metadata={"help": "Special tokens used in responses"}
    )
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.env_instruction:
            self.env_instruction = self._get_default_instruction()
    
    def _get_default_instruction(self) -> str:
        """Generate default instruction based on environment type"""
        instructions = {
            "cliffwalking": (
                "Navigate from start to goal while avoiding the cliff. "
                "Available actions: up, down, left, right. "
                "The cliff (marked as 'C') will cause you to fall and receive a large penalty."
            ),
            "gridworld": (
                "Navigate through the grid to reach the goal position. "
                "Available actions: up, down, left, right."
            ),
            "gobang_1v1": (
                "Play Gobang (Five in a Row). Place your pieces to get 5 consecutive pieces "
                "in a row, column, or diagonal. Format: 'row col' (e.g., '7 7')"
            ),
            "reversi_1v1": (
                "Play Reversi (Othello). Place pieces to capture opponent's pieces by "
                "flanking them. Format: 'row col'"
            ),
            "chinesechess": (
                "Play Chinese Chess using standard notation. "
                "Examples: '车1进3' (Rook column 1 forward 3), '马8进7' (Horse 8 to 7)"
            ),
            "snakes_1v1": (
                "Control your snake to eat food and avoid collisions. "
                "Available actions: up, down, left, right"
            ),
            "sokoban_1p": (
                "Push boxes to target positions in this puzzle game. "
                "You can only push boxes, not pull them. Available actions: up, down, left, right"
            ),
            "delivery_two_agents": (
                "Efficiently deliver packages to their destinations. "
                "Actions: move up/down/left/right, grab orders, deliver orders"
            ),
            "seek_2p": (
                "Multi-agent seek game. Find and catch targets while avoiding being caught. "
                "Available actions: up, down, left, right"
            )
        }
        return instructions.get(self.jidi_env_name, f"Play the {self.jidi_env_name} game.")


@dataclass  
class CliffWalkingConfig(JidiBaseConfig):
    """Configuration for CliffWalking environment"""
    
    jidi_env_name: str = field(default="cliffwalking")
    max_steps: int = field(default=200)
    max_tokens_per_step: int = field(default=64)
    
    env_instruction: str = field(
        default=(
            "You are navigating a cliff walking environment. "
            "Start at bottom-left, reach the goal at bottom-right. "
            "Avoid the cliff (marked as 'C') which gives -100 reward. "
            "Each step gives -1 reward, reaching goal gives 0 reward. "
            "Available actions: up, down, left, right."
        )
    )
    
    # Grid display configuration
    grid_height: int = field(default=4)
    grid_width: int = field(default=12)
    
    # Reward structure
    step_penalty: float = field(default=-1.0)
    cliff_penalty: float = field(default=-100.0)
    goal_reward: float = field(default=0.0)


@dataclass
class GridWorldConfig(JidiBaseConfig):
    """Configuration for GridWorld environment"""
    
    jidi_env_name: str = field(default="gridworld")
    max_steps: int = field(default=100)
    max_tokens_per_step: int = field(default=64)
    
    env_instruction: str = field(
        default=(
            "Navigate through the grid world to reach the goal position. "
            "Available actions: up, down, left, right. "
            "Plan your path efficiently to minimize steps."
        )
    )
    
    # Grid configuration
    grid_size: str = field(default="unknown")  # Will be determined at runtime
    step_penalty: float = field(default=-1.0)
    goal_reward: float = field(default=10.0)


@dataclass
class GobangConfig(JidiBaseConfig):
    """Configuration for Gobang (Five in a Row) environment"""
    
    jidi_env_name: str = field(default="gobang_1v1")
    max_steps: int = field(default=225)  # 15x15 board
    max_tokens_per_step: int = field(default=128)
    
    # Board configuration
    board_size: int = field(default=15)
    win_condition: int = field(default=5)


@dataclass
class ChineseChessConfig(JidiBaseConfig):
    """Configuration for Chinese Chess environment"""
    
    jidi_env_name: str = field(default="chinesechess")
    max_steps: int = field(default=400)
    max_tokens_per_step: int = field(default=256)
    
    # Chess specific settings
    time_limit: Optional[int] = field(default=None, metadata={"help": "Time limit per move in seconds"})


@dataclass
class MiniGridConfig(JidiBaseConfig):
    """Configuration for MiniGrid environment"""
    
    jidi_env_name: str = field(default="MiniGrid-DoorKey-16x16-v0")
    max_steps: int = field(default=300)
    max_tokens_per_step: int = field(default=128)
    
    env_instruction: str = field(
        default=(
            "Navigate through the MiniGrid environment to complete your objective. "
            "Available actions: forward, left, right, pickup, drop, toggle, done. "
            "Observe the environment carefully and plan your actions."
        )
    )
    
    # MiniGrid specific configuration
    grid_size: int = field(default=16)
    render_mode: str = field(default="text")


@dataclass
class SokobanConfig(JidiBaseConfig):
    """Configuration for Sokoban environment"""
    
    jidi_env_name: str = field(default="sokoban_1p") 
    max_steps: int = field(default=200)
    max_tokens_per_step: int = field(default=128)
    
    env_instruction: str = field(
        default=(
            "Solve the Sokoban puzzle by pushing all boxes to target positions. "
            "Available actions: up, down, left, right. "
            "You can only push boxes, not pull them. Plan carefully to avoid getting stuck."
        )
    )
    
    # Sokoban specific configuration
    puzzle_difficulty: str = field(default="medium")
    box_penalty: float = field(default=-0.1)  # Small penalty for each move
    target_reward: float = field(default=10.0)  # Reward for placing box on target


@dataclass
class SnakesConfig(JidiBaseConfig):
    """Configuration for Snakes environment"""
    
    jidi_env_name: str = field(default="snakes_1v1")
    max_steps: int = field(default=300)
    max_tokens_per_step: int = field(default=64)


@dataclass
class DeliveryConfig(JidiBaseConfig):
    """Configuration for Delivery environment"""
    
    jidi_env_name: str = field(default="delivery_two_agents")
    max_steps: int = field(default=200)
    max_tokens_per_step: int = field(default=128)


@dataclass
class ReversiConfig(JidiBaseConfig):
    """Configuration for Reversi environment"""
    
    jidi_env_name: str = field(default="reversi_1v1")
    max_steps: int = field(default=100)  # 10x10 board
    max_tokens_per_step: int = field(default=128)
    
    board_size: int = field(default=10)


@dataclass
class SeekConfig(JidiBaseConfig):
    """Configuration for Seek (multi-agent) environment"""
    
    jidi_env_name: str = field(default="seek_2p")
    max_steps: int = field(default=150)
    max_tokens_per_step: int = field(default=64)


# Environment name to config class mapping
JIDI_CONFIG_REGISTRY = {
    "cliffwalking": CliffWalkingConfig,
    "gridworld": GridWorldConfig,
    "minigrid": MiniGridConfig,
    "sokoban": SokobanConfig,
    "gobang": GobangConfig,
    "chinesechess": ChineseChessConfig,
    "snakes": SnakesConfig,
    "delivery": DeliveryConfig,
    "reversi": ReversiConfig,
    "seek": SeekConfig,
}
