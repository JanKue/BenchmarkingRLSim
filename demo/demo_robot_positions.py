import numpy as np
from alr_sim.core import RobotPlotFlags, RobotLogger
from alr_sim.sims.SimFactory import SimRepository
from gym.spaces import Box

if __name__ == '__main__':

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)
    scene.start()
    duration = 4

    controller = robot.torqueController

    robot.set_desired_gripper_width(0.02)

    original_qpos = scene.init_qpos
    print(original_qpos)

    variance = np.array([0.025, 0.025, 0.025, 0.025, 0.02, 0.02, 0.02, 0, 0])
    variance_space = Box(low=-variance, high=variance)

    for i in range(100):

        sample = variance_space.sample()
        new_qpos = original_qpos + sample
        scene.init_qpos = new_qpos

        print("sample", sample)
        print("new position", new_qpos)

        scene.reset()

        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        robot.executeTorqueCtrlTimeStep(action, timeSteps=1_000)

