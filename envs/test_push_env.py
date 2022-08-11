import time
import random
import numpy as np

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.sims.SimFactory import SimRepository

from alr_sim.core.logger import RobotPlotFlags

from push_env import PushEnv

import gym.spaces



if __name__ == "__main__":

    sim_factory = SimRepository.get_factory("mujoco")

    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)
    # ctrl = GymCartesianVelController(
    #     robot,
    #     fixed_orientation=np.array([0, 1, 0, 0]),
    #     max_cart_vel=0.1,
    #     use_spline=False,
    # )
    # robot.cartesianPosQuatTrackingController.neglect_dynamics = False
    ctrl = GymTorqueController(robot)
    env = PushEnv(scene, robot, ctrl, max_steps=500, random_env=False)

    env.start()

    scene.start_logging()
    env.reset()
    for i in range(200):
        action = ctrl.action_space().sample()
        print("action:", action)
        step_obs, step_reward, step_done, _ = env.step(action)
        print("%d: Position " % (i), robot.current_c_pos)
        print("Latest step reward:", step_reward)
        print("observation: ", step_obs)
        print("valid observation: ", env.observation_space.contains(step_obs))

        if step_done or (i + 1) % 50 == 0:
            print("reset!")
            env.reset()

    scene.stop_logging()
    robot.robot_logger.plot(
        plot_selection=RobotPlotFlags.JOINTS
        | RobotPlotFlags.END_EFFECTOR
        | RobotPlotFlags.MISC
    )
