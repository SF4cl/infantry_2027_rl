"""Terrain definitions used by infantry_2027 experiments."""

from .fudan_plane import (
    COLUMN_NAMES,
    FUDAN_TERRAINS_CFG,
    TRAINING_PROPORTIONS,
    TRAINING_VARIANTS,
    build_fudan_terrain_grid,
)

__all__ = [
    "COLUMN_NAMES",
    "FUDAN_TERRAINS_CFG",
    "TRAINING_PROPORTIONS",
    "TRAINING_VARIANTS",
    "build_fudan_terrain_grid",
]
