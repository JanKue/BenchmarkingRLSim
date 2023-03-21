from abc import ABC

import numpy as np
from alr_sim.core import Scene
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.sims.SimFactory import SimRepository

from objects.hammer_objects import HammerObjects

from gym.spaces import Box


class HammerEnv(GymEnvWrapper, ABC):
    def __init__(
        self,
        simulator: str = "mujoco",
        n_substeps: int = 10,
        max_steps_per_episode: int = 250,
        debug: bool = True,
        random_goal: bool = False,
        random_init: bool = False,
        render=False
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

        hammer_objects = HammerObjects(name="hammer_objects")
        scene.add_object(hammer_objects)

        self.nail_goal = 0.1
        self.success_threshold = 0.03

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64)
        self.action_space = self.controller.action_space()
        self.reward_range = (-np.inf, 0)

        self.start()

    def get_observation(self) -> np.ndarray:
        robot_state = self.robot_state()

        tcp_pos = self.robot.current_c_pos
        handle_pos = self.scene.sim.data.get_geom_xpos("HammerHandle")
        tcp_handle_distance = np.linalg.norm(tcp_pos - handle_pos)
        nail_joint_pos = self.scene.sim.data.get_joint_qpos("NailSlideJoint")
        nail_goal_difference = self.nail_goal - nail_joint_pos

        env_state = np.concatenate([tcp_pos, handle_pos, [tcp_handle_distance],
                                    [nail_joint_pos, self.nail_goal, nail_goal_difference]])

        return np.concatenate([env_state, robot_state])

    def get_reward(self):

        tcp_pos = self.robot.current_c_pos
        handle_pos = self.scene.sim.data.get_geom_xpos("HammerHandle")
        tcp_handle_distance = np.linalg.norm(tcp_pos - handle_pos)

        nail_joint_pos = self.scene.sim.data.get_joint_qpos("NailSlideJoint")
        nail_goal_difference = self.nail_goal - nail_joint_pos

        reward = - 10 * tcp_handle_distance - 100 * nail_goal_difference

        return reward

    def _check_early_termination(self) -> bool:
        nail_joint_pos = self.scene.sim.data.get_joint_qpos("NailSlideJoint")
        nail_goal_difference = self.nail_goal - nail_joint_pos
        if nail_goal_difference < self.success_threshold:
            self.terminated = True
            return True
        return False

    def _reset_env(self):
        self.scene.reset()

    def reset(self):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self._reset_env()
        return self.get_observation()

    def debug_msg(self) -> dict:
        nail_joint_pos = self.scene.sim.data.get_joint_qpos("NailSlideJoint")
        nail_goal_difference = self.nail_goal - nail_joint_pos
        success = nail_goal_difference < self.success_threshold
        return {"is_success": success}
