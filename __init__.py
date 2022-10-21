
from gym.envs.registration import register
from envs.reachenv import ReachEnv

register(
    id="ReachEnv-v0",
    entry_point="envs.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v1",
    entry_point="envs.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="ReachEnv-v2",
    entry_point="envs.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v3",
    entry_point="envs.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "simulator": 'mujoco',
            "render": True}
)