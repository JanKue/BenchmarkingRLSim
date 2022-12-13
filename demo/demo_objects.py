import numpy as np
from alr_sim.sims.SimFactory import SimRepository
import alr_sim.sims.universal_sim.PrimitiveObjects as PrimObj
from envs.objects.hammer_objects import HammerObjects
from envs.objects.door_objects import DoorObjects

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
        static=True,
        visual_only=True
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

    hammer_objects = HammerObjects(
        name="hammer-and-block",
        init_pos=[0.0, 0.0, 0.0],
        init_quat=[0.0, 0.0, 0.0, 0.0]
    )

    door_objects = DoorObjects(
        name="safe-and-door",
        init_pos=[0.0, 0.0, 0.0],
        init_quat=[0.0, 0.0, 0.0, 0.0]
    )

    object_list = [door_objects, cylinder2]

    # SCENE SETUP

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=object_list, random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()
    duration = 4

    # MOVEMENT

    robot.set_desired_gripper_width(0.01)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat

    # DOOR DEMO

    print(scene.sim.data.get_geom_xpos("hand_target"))
    print(scene.sim.data.get_body_xpos("doorlockB"))
    print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.6, 0.4, 0.1], [0, 1, 1, 0], duration=duration
    )

    target_pos = scene.get_obj_pos(cylinder2)
    tcp_pos = robot.current_c_pos
    print(target_pos)
    print(tcp_pos)
    pos_diff = np.abs(target_pos - tcp_pos)
    print(pos_diff)
    print(pos_diff < [0.08, 0.08, 0.025])
    print(np.all(pos_diff < [0.08, 0.08, 0.025]))


    # print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.15, 0.1, 0.1], [0, 1, 1, 0], duration=duration
    )

    # print(scene.sim.data.get_geom_xpos("handle"))
    # print(scene.sim.data.get_joint_qpos("doorjoint"))

    scene.reset()
    # print(scene.sim.data.get_geom_xpos("handle"))
    # print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.6, 0.39, 0.1], [0, 1, 1, 0], duration=duration
    )

    # HAMMER DEMO

    # print(scene.sim.data.get_joint_qpos("NailSlideJoint"))
    #
    # robot.set_desired_gripper_width(0.025)
    #
    # robot.gotoCartPositionAndQuat(
    #     [0.5, 0.5, 0.0], [0, 0, 0, 0], duration=duration
    # )
    #
    # robot.set_desired_gripper_width(0.005)
    #
    # robot.gotoCartPositionAndQuat(
    #     [0.5, -0.5, 0.1], [0, 0, 0, 0], duration=duration
    # )
    #
    # print(scene.sim.data.get_joint_qpos("NailSlideJoint"))

    # hammerhead = scene.get_object("hammerhead")
    # print(scene.get_obj_pos(hammerhead))

