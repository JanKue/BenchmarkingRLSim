
from gym.envs.registration import register
from envs.reachenv import ReachEnv

register(
    id="ReachEnv-v0",
    entry_point="envs.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_env": False,
            "simulator": 'mujoco',
            "render": True}
)