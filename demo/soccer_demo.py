from alr_sim.sims.SimFactory import SimRepository
from envs.objects.soccer_objects import SoccerObjects

if __name__ == '__main__':

    # SCENE SETUP

    soccer_objects = SoccerObjects(name="soccer-ball-and-goal")

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=[soccer_objects], random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()

    # MOVEMENT SETUP

    robot.set_desired_gripper_width(0)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat
    duration = 2

    # DEMO

    print("ball pos", scene.sim.data.get_body_xpos("soccer_ball"))
    print("goal pos", scene.sim.data.get_body_xpos("soccer_goal"))

    robot.gotoCartPositionAndQuat(
        [0.4, -0.15, 0.01], [0, 1, 1, 0], duration=duration
    )

    robot.gotoCartPositionAndQuat(
        [0.4, -0.05, 0.01], [0, 1, 1, 0], duration=duration
    )

    print("ball pos", scene.sim.data.get_body_xpos("soccer_ball"))

    robot.gotoCartPositionAndQuat(
        init_pos, init_or, duration=duration
    )

    print("ball pos", scene.sim.data.get_body_xpos("soccer_ball"))
