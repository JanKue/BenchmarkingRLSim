import time

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

    object_list = [door_objects]

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

    print(scene.sim.data.get_geom_xpos("handle"))
    print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.5, 0.4, 0.1], [0, 1, 1, 0], duration=duration
    )

    print(scene.sim.data.get_joint_qpos("doorjoint"))

    robot.gotoCartPositionAndQuat(
        [0.15, 0.1, 0.1], [0, 1, 1, 0], duration=duration
    )

    print(scene.sim.data.get_geom_xpos("handle"))
    print(scene.sim.data.get_joint_qpos("doorjoint"))

    # scene.reset()
    # print(scene.sim.data.get_geom_xpos("handle"))
    # print(scene.sim.data.get_joint_qpos("doorjoint"))
    #
    # robot.gotoCartPositionAndQuat(
    #     [0.5, 0.4, 0.1], [0, 1, 1, 0], duration=duration
    # )

    # HAMMER DEMO

    # hammerhead = scene.get_object("hammerhead")
    # print(scene.get_obj_pos(hammerhead))

