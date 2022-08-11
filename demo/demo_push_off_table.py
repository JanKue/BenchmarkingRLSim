from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box, Cylinder

if __name__ == '__main__':

    box1 = Box(
        name="box1",
        init_pos=[0.6, 0.0, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1],
    )

    cylinder1 = Cylinder(
        name="cylinder1",
        init_pos=[0.6, -0.2, 0.50],
        init_quat=[0, 1, 0, 0],
        rgba=[0, 1, 0, 1],
    )

    table = Box(
        name="table0",
        init_pos=[0.5, 0.0, 0.2],
        init_quat=[0, 1, 0, 0],
        size=[0.25, 0.35, 0.2],
        static=True,
    )

    object_list = [box1, table, cylinder1]

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=object_list)
    robot = sim_factory.create_robot(scene)
    scene.start()
    duration = 2

    robot.set_desired_gripper_width(0.0)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat

    robot.gotoCartPositionAndQuat(
        [0.55, 0.0, 0.4], [0, 1, 0, 0], duration=duration
    )

    robot.gotoCartPositionAndQuat(
        [0.75, 0.0, 0.35], [0, 1, 0, 0], duration=duration/2
    )

    robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)