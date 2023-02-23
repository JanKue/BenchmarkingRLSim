from alr_sim.sims.SimFactory import SimRepository
import alr_sim.sims.universal_sim.PrimitiveObjects as PrimObj

if __name__ == '__main__':

    # OBJECT DEFINITION

    box1 = PrimObj.Box(
        name="box1",
        init_pos=[0.6, -0.2, 0.45],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1],
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

    sphere1 = PrimObj.Sphere(
        name="sphere1",
        init_pos=[0.55, 0.34, 0.135],
        init_quat=[0, 0, 0, 0],
        rgba=[0.3, 0.0, 0.0, 1],
        size=[0.09],
        mass=1.0,
        static=False,
        visual_only=False
    )

    sphere2 = PrimObj.Sphere(
        name="sphere2",
        init_pos=[0.49, 0.444, 0.135],
        init_quat=[0, 0, 0, 0],
        rgba=[0.3, 0.0, 0.0, 1],
        size=[0.02],
        mass=1.0,
        static=True,
        visual_only=True
    )

    cylinder2 = PrimObj.Cylinder(
        name="cylinder2",
        init_pos=[0.57, 0.39, 0.13],
        init_quat=[0, 1, 0, 0],
        rgba=[0.3, 0.0, 0.0, 1],
        size=[0.08, 0.025],
        mass=1.0,
        static=True,
        visual_only=True
    )

    object_list = [box1, cylinder1, table, sphere1, sphere2]

    # SCENE SETUP

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=object_list, random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()

    # MOVEMENT

    robot.set_desired_gripper_width(0.01)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat
    duration = 4

    robot.gotoCartPositionAndQuat(
        [0.4, -0.15, 0.01], [0, 1, 1, 0], duration=duration
    )