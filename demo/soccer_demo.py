from alr_sim.sims.SimFactory import SimRepository
import alr_sim.sims.universal_sim.PrimitiveObjects as PrimObj
from envs.objects.soccer_objects import SoccerObjects

if __name__ == '__main__':

    # SCENE SETUP

    sphere = PrimObj.Sphere(
        name="sphere",
        init_pos=[0.5, 0.54, 0.13],
        init_quat=[0, 0, 0, 0],
        rgba=[0.3, 0.0, 0.0, 1],
        size=[0.01],
        mass=1.0,
        static=True,
        visual_only=True
    )

    ball = PrimObj.Sphere(
        name="ball",
        init_pos=[0.6, 0.3, 0.01],
        init_quat=[0, 0, 0, 0],
        rgba=[0.8, 0.8, 0.8, 1],
        size=[0.026],
        mass=0.04,
        static=False,
        visual_only=False
    )

    soccer_objects = SoccerObjects(name="soccer-objects")

    # goal line: y = 0.44, back: y = 0.54
    # posts: x = 0.3, 0.5
    # bottom: z = -0.02, top: z = 0.13


    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=[soccer_objects, ball], random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()

    # MOVEMENT SETUP

    robot.set_desired_gripper_width(0)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat
    duration = 2

    # DEMO

    # print("ball pos", scene.sim.data.get_body_xpos("soccer_ball"))
    # print("goal pos", scene.sim.data.get_body_xpos("soccer_goal"))

    robot.gotoCartPositionAndQuat(
        [0.4, -0.15, 0.01], [0, 1, 1, 0], duration=duration
    )

    robot.gotoCartPositionAndQuat(
        [0.4, -0.05, 0.01], [0, 1, 1, 0], duration=duration
    )

    # print("ball pos", scene.sim.data.get_body_xpos("soccer_ball"))

    robot.gotoCartPositionAndQuat(
        init_pos, init_or, duration=duration
    )

    # print("ball pos", scene.sim.data.get_body_xpos("soccer_ball"))
