from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

from stable_baselines3 import SAC

env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]()

model = SAC("MlpPolicy", env=env, verbose=1)

model.learn(total_timesteps=500)

print(env.dt)

