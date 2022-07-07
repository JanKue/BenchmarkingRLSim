from alr_sim.sims.SimFactory import SimRepository

if __name__ == '__main__':

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)
    scene.start()
    duration = 4

    robot.set_desired_gripper_width(0.02)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat

    robot.gotoCartPositionAndQuat(
        [0.5, 0.0, 0.5], [0, 0, 1, 1], duration=duration
    )

    robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)