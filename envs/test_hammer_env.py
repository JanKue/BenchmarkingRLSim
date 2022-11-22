import time
import random
import numpy as np

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.sims.SimFactory import SimRepository

from alr_sim.core.logger import RobotPlotFlags

from meta_push_env import MetaPushEnv

import __init__

import gym.spaces



if __name__ == "__main__":

    env = gym.make("HammerEnv-v0")

    scene = env.scene
    robot = env.robot
    ctrl = env.controller

    env.reset()
    for i in range(10_000):
        action = ctrl.action_space().sample()
        # print("action:", action)
        step_obs, step_reward, step_done, _ = env.step(action)
        # print("%d: Position " % (i), robot.current_c_pos)
        print("Latest step reward:", step_reward)
        # print("observation: ", step_obs)
        # print("valid observation: ", env.observation_space.contains(step_obs))

        if step_done or (i + 1) % 250 == 0:
            print("reset!")
            env.reset()

