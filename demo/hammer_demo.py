from alr_sim.sims.SimFactory import SimRepository
from envs.objects.hammer_objects import HammerObjects

if __name__ == '__main__':

    # SCENE SETUP

    hammer_objects = HammerObjects(name="hammer-and-block")

    sim_factory = SimRepository.get_factory("mujoco")
    scene = sim_factory.create_scene(object_list=[hammer_objects], random_env=True)
    robot = sim_factory.create_robot(scene)
    scene.start()

    # MOVEMENT SETUP

    robot.set_desired_gripper_width(0.01)
    init_pos = robot.current_c_pos
    init_or = robot.current_c_quat
    duration = 4

    # DEMO

    print(scene.sim.data.get_joint_qpos("NailSlideJoint"))

    robot.set_desired_gripper_width(0.025)

    print("picking up hammer")
    robot.gotoCartPositionAndQuat(
        [0.5, 0.5, 0.0], [0, 0, 0, 0], duration=duration
    )
    print("picked up hammer")

    robot.set_desired_gripper_width(0.005)

    print("moving hammer")
    robot.gotoCartPositionAndQuat(
        [0.35, -0.25, 0.1], [0, 0, 0, 0], duration=duration
    )
    print("moved hammer")

    print("hitting nail")
    robot.gotoCartPositionAndQuat(
        [0.35, -0.4, 0.1], [0, 0, 0, 0], duration=duration
    )
    print("hit nail")

    print(scene.sim.data.get_joint_qpos("NailSlideJoint"))

    hammerhead = scene.get_object("hammerhead")
    print(scene.get_obj_pos(hammerhead))
