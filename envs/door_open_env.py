import numpy as np
from alr_sim.core import Scene
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.sims.SimFactory import SimRepository

from envs.objects.door_objects import DoorObjects


class DoorOpenEnv(GymEnvWrapper):
    def __init__(
        self,
        simulator: str,
        n_substeps: int = 10,
        max_steps_per_episode: int = 250,
        debug: bool = True,
        random_goal: bool = False,
        random_init: bool = False,
        render=False
    ):
        sim_factory = SimRepository.get_factory(simulator)
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        door_objects = DoorObjects(name="door_objects")
        scene = sim_factory.create_scene(render=render_mode, object_list=[door_objects])
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

    def get_observation(self) -> np.ndarray:
        return None

    def get_reward(self):
        return None

    def _check_early_termination(self) -> bool:
        return None

    def _reset_env(self):
        return None

    def reset(self):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self._reset_env()
        return self.get_observation()

    def debug_msg(self) -> dict:
        return None


