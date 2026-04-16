"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""
from roll.utils.logging import get_logger

# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
# from .bandit.config import BanditEnvConfig
# from .bandit.env import BanditEnv
# from .countdown.config import CountdownEnvConfig
# from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv as OriginalSokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
# from .metamathqa.env import MetaMathQAEnv
# from .metamathqa.config import MetaMathQAEnvConfig

# Jidi environments
from .jidi.config import CliffWalkingConfig, GridWorldConfig, MiniGridConfig, SokobanConfig
from .jidi.env import CliffWalkingEnv, GridWorldEnv, MiniGridEnv, SokobanEnv

# Search environment
from .search.config import SearchEnvConfig
from .search.env import SearchEnv

# NQ Search environment
from .nq_search.config import NQSearchEnvConfig
from .nq_search.env import NQSearchEnv

logger = get_logger()

REGISTERED_ENVS = {
    # "bandit": BanditEnv,
    # "countdown": CountdownEnv,
    "sokoban": OriginalSokobanEnv,
    "frozen_lake": FrozenLakeEnv,
    # 'alfworld': AlfredTXTEnv,
    # "metamathqa": MetaMathQAEnv,
    # Jidi environments
    "jidi_cliffwalking": CliffWalkingEnv,
    "jidi_gridworld": GridWorldEnv,
    "jidi_minigrid": MiniGridEnv,
    "jidi_sokoban": SokobanEnv,
    # Search environment
    "search": SearchEnv,
    # NQ Search environment
    "nq_search": NQSearchEnv,
}

REGISTERED_ENV_CONFIGS = {
    # "bandit": BanditEnvConfig,
    # "countdown": CountdownEnvConfig,
    "sokoban": SokobanEnvConfig,
    "frozen_lake": FrozenLakeEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    # "metamathqa": MetaMathQAEnvConfig,
    # Jidi environments
    "jidi_cliffwalking": CliffWalkingConfig,
    "jidi_gridworld": GridWorldConfig,
    "jidi_minigrid": MiniGridConfig,
    "jidi_sokoban": SokobanConfig,
    # Search environment
    "search": SearchEnvConfig,
    # NQ Search environment
    "nq_search": NQSearchEnvConfig,
}

# Webshop environment disabled due to missing dependencies
# try:
#     # add webshop-minimal to PYTHONPATH
#     import os
#     import sys
# 
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     relative_path = "../../../third_party/webshop-minimal"
#     module_path = os.path.join(current_dir, relative_path)
#     sys.path.append(module_path)
# 
#     from .webshop.config import WebShopEnvConfig
#     from .webshop.env import WebShopEnv
# 
#     REGISTERED_ENVS["webshop"] = WebShopEnv
#     REGISTERED_ENV_CONFIGS["webshop"] = WebShopEnvConfig
# except Exception as e:
#     logger.info(f"Failed to import webshop: {e}")
