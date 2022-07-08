import numpy as np
import random
from collections import namedtuple, deque

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

from alr_sim.gyms.gym_controllers import GymCartesianVelController
from alr_sim.sims.SimFactory import SimRepository
from envs.reach_env.reach import ReachEnv
from modified_reach_env import ModReachEnv

from alr_sim.core.logger import RobotPlotFlags

if __name__ == "__main__":

    # create scene and environment

    sim_factory = SimRepository.get_factory("mujoco")

    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)
    ctrl = GymCartesianVelController(
        robot,
        fixed_orientation=np.array([0, 1, 0, 0]),
        max_cart_vel=0.1,
        use_spline=False,
    )
    robot.cartesianPosQuatTrackingController.neglect_dynamics = False
    env = ModReachEnv(scene=scene, n_substeps=500, controller=ctrl, random_env=False)

    env.start()
    env.seed(1)
    scene.start_logging()

    # create model

    check_env(env)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100)
    model.save("a2c_reach_model")

    # run training

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print("action", action, "observation", obs, "reward", rewards)
        if done:
            obs = env.reset()