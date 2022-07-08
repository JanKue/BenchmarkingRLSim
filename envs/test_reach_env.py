import time

import numpy as np

from alr_sim.gyms.gym_controllers import GymCartesianVelController
from alr_sim.sims.SimFactory import SimRepository
from envs.reach_env.reach import ReachEnv
from modified_reach_env import ModReachEnv

from alr_sim.core.logger import RobotPlotFlags

import random


if __name__ == "__main__":

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
    env = ModReachEnv(scene=scene, n_substeps=500, controller=ctrl, random_env=True, debug=True)

    env.start()

    env.seed(10)
    scene.start_logging()
    for i in range(20):
        action = ctrl.action_space().sample()
        print(action)
        step_obs, step_reward, _, _ = env.step(action)
        print("%d: State " % (i), robot.current_c_pos, ctrl.desired_c_pos)
        print("Latest step reward:", step_reward)
        print("latest observation", step_obs)
        if (i + 1) % 100 == 0:
            env.reset()

    scene.stop_logging()
    robot.robot_logger.plot(
        plot_selection=RobotPlotFlags.JOINTS
        | RobotPlotFlags.END_EFFECTOR
        | RobotPlotFlags.MISC
    )
