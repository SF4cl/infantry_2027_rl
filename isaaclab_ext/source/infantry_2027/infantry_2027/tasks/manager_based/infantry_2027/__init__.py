import gymnasium as gym

from . import agents


gym.register(
    id="Infantry-2027-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_env_cfg:Infantry2027FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027PPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_terrain_env_cfg:Infantry2027TerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainPPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Terrain-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_terrain_env_cfg:Infantry2027TerrainPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainPPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_env_cfg:Infantry2027FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027PPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Flat-Compatible-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027FlatCompatibleEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027FlatCompatiblePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Flat-Compatible-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027FlatCompatiblePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027FlatCompatiblePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Terrain-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027TerrainV1EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainV1PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Terrain-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027TerrainV1PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainV1PPORunnerCfg"
        ),
    },
)
