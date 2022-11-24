from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]()

model = SAC("MlpPolicy", env=env, verbose=1)

model.learn(total_timesteps=750)

