import numpy as np
from alr_sim.core import Scene
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.sims.SimFactory import SimRepository

from envs.objects.door_objects import DoorObjects

from gym.spaces import Box


class DoorOpenEnv(GymEnvWrapper):
    def __init__(
        self,
        simulator: str,
        n_substeps: int = 10,
        max_steps_per_episode: int = 625,
        debug: bool = True,
        random_init: bool = False,
        render=True
    ):
        sim_factory = SimRepository.get_factory(simulator)
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(render=render_mode)
        robot = sim_factory.create_robot(scene)
        controller = GymTorqueController(robot)
        robot.cartesianPosQuatTrackingController.neglect_dynamics = False
        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        door_objects = DoorObjects(name="door_objects")
        scene.add_object(door_objects)

        self.hinge_goal = -np.pi/2
        self.success_threshold = np.pi/6

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64)
        self.action_space = self.controller.action_space()
        self.reward_range = (-np.inf, 0)

        self.start()

    def get_observation(self) -> np.ndarray:
        robot_state = self.robot_state()

        tcp_pos = self.robot.current_c_pos
        handle_pos = self.scene.sim.data.get_geom_xpos("handle")
        tcp_handle_distance = np.linalg.norm(tcp_pos - handle_pos)
        hand_target_pos = self.scene.sim.data.get_geom_xpos("hand_target")
        hand_target_diff = tcp_pos - hand_target_pos
        hand_target_distance = np.linalg.norm(hand_target_diff)
        cylinder_size = [0.08, 0.08, 0.025]
        hinge_pos = self.scene.sim.data.get_joint_qpos("doorjoint")
        hinge_difference = hinge_pos - self.hinge_goal

        env_state = np.concatenate([tcp_pos, handle_pos, [tcp_handle_distance],
                                    hand_target_pos, hand_target_diff, [hand_target_distance], cylinder_size,
                                    [hinge_pos, self.hinge_goal, hinge_difference]])

        return np.concatenate([robot_state, env_state])

    def get_reward(self):
        self.robot.receiveState()
        tcp_pos = self.robot.current_c_pos
        tcp_quat = self.robot.current_c_quat

        # calculate distance between robot tcp and door hinge
        handle_pos = self.scene.sim.data.get_geom_xpos("handle")
        tcp_handle_distance = np.linalg.norm(tcp_pos - handle_pos)

        # guide the end effector to a cylinder between handle and door
        hand_target_pos = self.scene.sim.data.get_geom_xpos("hand_target")
        hand_target_diff = tcp_pos - hand_target_pos
        hand_target_distance = np.linalg.norm(hand_target_diff)
        cylinder_size = [0.08, 0.08, 0.025]
        hand_in_cylinder = np.abs(hand_target_diff) < cylinder_size

        # calculate door opening angle and compare to target value
        hinge_pos = self.scene.sim.data.get_joint_qpos("doorjoint")
        hinge_difference = hinge_pos - self.hinge_goal
        exp_hinge_diff = np.exp(hinge_difference) - 1.0

        # keep robot end-effector upright by keeping W and Z close to 0
        w_error = (np.exp(abs(tcp_quat[0])) - 1) ** 2
        z_error = (np.exp(abs(tcp_quat[3])) - 1) ** 2
        orientation_error = np.minimum(w_error + z_error, 10)

        # if tcp_handle_distance > 0.075:
        #     reward = - 5 * tcp_handle_distance - 20 * hinge_difference
        # else:
        #     reward = - 20 * hinge_difference

        reward = - 10 * exp_hinge_diff - 2 * orientation_error

        if not np.all(hand_in_cylinder):
            reward -= 100 * hand_target_distance

        return np.minimum(reward, 0)

    def _check_early_termination(self) -> bool:
        hinge_pos = self.scene.sim.data.get_joint_qpos("doorjoint")
        hinge_difference = hinge_pos - self.hinge_goal
        # terminates if door hinge is opened far enough
        if hinge_difference < self.success_threshold:
            self.terminated = True
            return True
        return False

    def _reset_env(self):
        self.scene.reset()

    def debug_msg(self) -> dict:
        hinge_pos = self.scene.sim.data.get_joint_qpos("doorjoint")
        hinge_difference = hinge_pos - self.hinge_goal
        success = hinge_difference < self.success_threshold
        return {"is_success": success}


