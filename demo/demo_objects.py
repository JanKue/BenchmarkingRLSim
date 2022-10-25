from alr_sim.sims.SimFactory import SimRepository
import alr_sim.sims.universal_sim.PrimitiveObjects as PrimObj
from objects.hammer import Hammer

if __name__ == '__main__':

    box1 = PrimObj.Box(
        name="box1",
        init_pos=[0.6, -0.2, 0.45],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1],
    )

    sphere1 = PrimObj.Sphere(
        name="sphere1",
        init_pos=[0.6, 0.0, 0.45],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1]
    )

    cylinder1 = PrimObj.Cylinder(
        name="cylinder1",
        init_pos=[0.6, 0.2, 0.45],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1]
    )

    table = PrimObj.Box(
        name="table0",
        init_pos=[0.5, 0.0, 0.2],
        init_quat=[0, 1, 0, 0],
        size=[0.25, 0.35, 0.2],
        static=True,
    )

    hammer = Hammer(
        name="hammer0",
        init_pos=[0.0, 0.0, 0.0],
        init_quat=[0.0, 0.0, 0.0, 0.0]
    )

    object_list = [table]

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=object_list, random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()
    duration = 4

    robot.set_desired_gripper_width(0.02)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat

    print(init_pos)
    print(init_or)

    robot.gotoCartPositionAndQuat(
        [0.5, 0.0, 0.5], init_or, duration=duration
    )

    robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)

