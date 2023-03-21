import numpy as np
from alr_sim.sims.SimFactory import SimRepository
import alr_sim.sims.universal_sim.PrimitiveObjects as PrimObj
from objects.door_objects import DoorBox

if __name__ == '__main__':

    # SCENE SETUP

    cylinder = PrimObj.Cylinder(
        name="cylinder",
        init_pos=[0.57, 0.39, 0.13],
        init_quat=[0, 1, 0, 0],
        rgba=[0.3, 0.0, 0.0, 1],
        size=[0.08, 0.025],
        mass=1.0,
        static=True,
        visual_only=True
    )

    door_objects = DoorBox(name="safe-and-door")
    object_list = [door_objects, cylinder]

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=object_list, random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()

    # MOVEMENT SETUP

    robot.set_desired_gripper_width(0)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat
    duration = 4

    # DEMO

    print(scene.sim.data.get_geom_xpos("hand_target"))
    print(scene.sim.data.get_body_xpos("doorlockB"))
    print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.6, 0.4, 0.1], [0, 1, 1, 0], duration=duration
    )

    target_pos = scene.get_obj_pos(cylinder)
    tcp_pos = robot.current_c_pos
    print(target_pos)
    print(tcp_pos)
    pos_diff = np.abs(target_pos - tcp_pos)
    print(pos_diff)
    print(pos_diff < [0.08, 0.08, 0.025])
    print(np.all(pos_diff < [0.08, 0.08, 0.025]))


    print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.15, 0.1, 0.1], [0, 1, 1, 0], duration=duration
    )

    print(scene.sim.data.get_geom_xpos("handle"))
    print(scene.sim.data.get_joint_qpos("doorjoint"))

    scene.reset()
    print(scene.sim.data.get_geom_xpos("handle"))
    print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.6, 0.39, 0.1], [0, 1, 1, 0], duration=duration
    )