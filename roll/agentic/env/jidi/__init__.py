"""
Jidi Environment Integration for ROLL

This package provides integration between Jidi reinforcement learning environments
and the ROLL training framework. It allows ROLL to utilize Jidi's diverse set of
environments including:

- Navigation environments (CliffWalking, GridWorld)
- Board games (Gobang, Chinese Chess, Reversi)
- Puzzle games (Sokoban)
- Multi-agent environments (Snakes, Delivery, Seek)

The integration leverages Jidi's existing LLM interactors for text-based interaction
while maintaining compatibility with ROLL's training pipeline.
"""

from .config import JidiBaseConfig, CliffWalkingConfig, GridWorldConfig, MiniGridConfig, SokobanConfig
from .env import JidiBaseEnv, CliffWalkingEnv, GridWorldEnv, MiniGridEnv, SokobanEnv

__all__ = [
    'JidiBaseConfig',
    'CliffWalkingConfig',
    'GridWorldConfig', 
    'MiniGridConfig',
    'SokobanConfig',
    'JidiBaseEnv',
    'CliffWalkingEnv',
    'GridWorldEnv',
    'MiniGridEnv', 
    'SokobanEnv',
]
