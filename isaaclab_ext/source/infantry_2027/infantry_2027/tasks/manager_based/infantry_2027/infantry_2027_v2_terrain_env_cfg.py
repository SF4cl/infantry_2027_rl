"""Stable flat-to-terrain continuation for the verified v2 VMC policy."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .infantry_2027_env_cfg import EventCfg, RewardsCfg
from .infantry_2027_terrain_env_cfg import (
    TERRAIN_PENALIZED_CONTACT_BODIES,
    Infantry2027TerrainSceneCfg,
    TerrainCurriculumCfg,
    TerrainObservationsCfg,
)
from .infantry_2027_v1_env_cfg import _disable_randomization
from .infantry_2027_v2_env_cfg import Infantry2027FlatStableEnvCfg


@configclass
class StableTerrainCommandsCfg:
    """Five-second commands with aligned forward/reverse terrain traversal."""

    motion = mdp.TerrainTraversalCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        flat_terrain_types=tuple(range(10)),
    )


@configclass
class StableTerrainRewardsCfg(RewardsCfg):
    """Preserve the accepted flat rewards and change only terrain semantics."""

    base_height = RewTerm(
        func=mdp.base_height,
        weight=1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    collision = RewTerm(
        func=mdp.collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=TERRAIN_PENALIZED_CONTACT_BODIES
            ),
            "threshold": 0.1,
        },
    )


@configclass
class Infantry2027TerrainStableEnvCfg(Infantry2027FlatStableEnvCfg):
    """Terrain v2 with a checkpoint-identical 125-D actor and 145-D critic."""

    scene: Infantry2027TerrainSceneCfg = Infantry2027TerrainSceneCfg(
        num_envs=1024, env_spacing=3.0
    )
    commands: StableTerrainCommandsCfg = StableTerrainCommandsCfg()
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    rewards: StableTerrainRewardsCfg = StableTerrainRewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # A mature flat policy must earn access to harder terrain instead of
        # being scattered over levels 0--5 as in the legacy v0 task.
        self.scene.terrain.max_init_terrain_level = 0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class Infantry2027TerrainStablePlayEnvCfg(Infantry2027TerrainStableEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.curriculum_steps = 0
        _disable_randomization(self)
