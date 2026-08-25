"""Faithful preview grid of every terrain branch in Fudan ``plane``.

The reference uses 8 m tiles, 0.1 m horizontal samples, 0.005 m vertical
samples and difficulties ``row / 10``.  Its signed slope branches are expanded
into separate columns here so every scene can be inspected at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.interpolate as interpolate
import trimesh

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass


TILE_SIZE = 8.0
HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
SLOPE_THRESHOLD = 0.75
NUM_LEVELS = 10

COLUMN_NAMES = (
    "flat",
    "smooth_slope_positive",
    "smooth_slope_negative",
    "rough_slope_positive",
    "rough_slope_negative",
    "stairs_negative",
    "stairs_positive",
    "discrete_obstacles",
    "stepping_stones",
    "gap",
    "pit",
)

# The final terrain checkpoint in the reference repository only trained these
# seven visible variants.  The weights resolve to exactly 20 curriculum
# columns, preserving the reference's final 50/20/10/10/10 distribution.
TRAINING_VARIANTS = (
    "flat",
    "smooth_slope_positive",
    "smooth_slope_negative",
    "rough_slope_positive",
    "rough_slope_negative",
    "stairs_negative",
    "stairs_positive",
)
TRAINING_PROPORTIONS = (0.50, 0.10, 0.10, 0.05, 0.05, 0.10, 0.10)


@dataclass(frozen=True)
class FudanTerrainGrid:
    mesh: trimesh.Trimesh
    origins: np.ndarray
    difficulties: np.ndarray
    column_names: tuple[str, ...]


def _pyramid_slope(difficulty: float, sign: float, rough: bool, rng: np.random.Generator) -> np.ndarray:
    count = int(TILE_SIZE / HORIZONTAL_SCALE)
    slope = sign * difficulty * (0.25 if rough else 0.5)
    height_max = slope * TILE_SIZE * 0.5 / VERTICAL_SCALE
    center = count // 2
    axis = (center - np.abs(center - np.arange(count))) / center
    field = height_max * axis[:, None] * axis[None, :]
    platform_half = int(3.0 / HORIZONTAL_SCALE / 2)
    platform_height = field[center - platform_half, center - platform_half]
    field = np.clip(field, min(0.0, platform_height), max(0.0, platform_height))
    if rough:
        amplitude = 0.05 + difficulty * 0.05
        downsampled_scale = 0.2
        down_count = int(TILE_SIZE / downsampled_scale)
        possible = np.arange(-amplitude, amplitude + 0.005, 0.005) / VERTICAL_SCALE
        low_res = rng.choice(np.rint(possible).astype(np.int16), size=(down_count, down_count))
        source = np.linspace(0.0, TILE_SIZE, down_count)
        target = np.linspace(0.0, TILE_SIZE, count)
        field += interpolate.RectBivariateSpline(source, source, low_res)(target, target)
    return np.rint(field).astype(np.int16)


def _stairs(difficulty: float, sign: float) -> np.ndarray:
    count = int(TILE_SIZE / HORIZONTAL_SCALE)
    step_width = int(0.7 / HORIZONTAL_SCALE)
    step_height = int(sign * (0.05 + 0.18 * difficulty) / VERTICAL_SCALE)
    platform_width = int(4.0 / HORIZONTAL_SCALE)
    field = np.zeros((count, count), dtype=np.int16)
    current_height = 0
    start = 0
    stop = count
    while stop - start > platform_width:
        start += step_width
        stop -= step_width
        current_height += step_height
        field[start:stop, start:stop] = current_height
    return field


def _discrete_obstacles(difficulty: float, rng: np.random.Generator) -> np.ndarray:
    count = int(TILE_SIZE / HORIZONTAL_SCALE)
    max_height = int((0.05 + difficulty * 0.1) / VERTICAL_SCALE)
    max_size = int(2.0 / HORIZONTAL_SCALE)
    widths = np.arange(int(1.0 / HORIZONTAL_SCALE), max_size, 4)
    positions = np.arange(0, count - max_size, 4)
    heights = np.asarray((-max_height, -max_height / 2, max_height / 2, max_height), dtype=np.int16)
    field = np.zeros((count, count), dtype=np.int16)
    for _ in range(20):
        width = int(rng.choice(widths))
        length = int(rng.choice(widths))
        x = int(rng.choice(positions))
        y = int(rng.choice(positions))
        field[x : x + width, y : y + length] = int(rng.choice(heights))
    platform = int(3.0 / HORIZONTAL_SCALE)
    start = (count - platform) // 2
    field[start : start + platform, start : start + platform] = 0
    return field


def _stepping_stones(difficulty: float, rng: np.random.Generator) -> np.ndarray:
    count = int(TILE_SIZE / HORIZONTAL_SCALE)
    stone_width = int((1.5 * (1.05 - difficulty)) / HORIZONTAL_SCALE)
    stone_width = max(1, stone_width)
    distance = int((0.05 if difficulty == 0.0 else 0.1) / HORIZONTAL_SCALE)
    # Reference max_height is zero. Its integer sampler consequently produces
    # stones at -1 height unit, i.e. -5 mm, over a -10 m hole field.
    field = np.full((count, count), int(-10.0 / VERTICAL_SCALE), dtype=np.int16)
    start_y = 0
    while start_y < count:
        stop_y = min(count, start_y + stone_width)
        start_x = int(rng.integers(0, stone_width))
        stop_x = max(0, start_x - distance)
        field[0:stop_x, start_y:stop_y] = -1
        while start_x < count:
            stop_x = min(count, start_x + stone_width)
            field[start_x:stop_x, start_y:stop_y] = -1
            start_x += stone_width + distance
        start_y += stone_width + distance
    platform = int(4.0 / HORIZONTAL_SCALE)
    start = (count - platform) // 2
    field[start : start + platform, start : start + platform] = 0
    return field


def _gap(difficulty: float) -> np.ndarray:
    count = int(TILE_SIZE / HORIZONTAL_SCALE)
    field = np.zeros((count, count), dtype=np.int16)
    gap_size = int(difficulty / HORIZONTAL_SCALE)
    platform_size = int(3.0 / HORIZONTAL_SCALE)
    center = count // 2
    inner = (count - platform_size) // 2
    outer = inner + gap_size
    field[center - outer : center + outer, center - outer : center + outer] = -1000
    field[center - inner : center + inner, center - inner : center + inner] = 0
    return field


def _pit(difficulty: float) -> np.ndarray:
    count = int(TILE_SIZE / HORIZONTAL_SCALE)
    field = np.zeros((count, count), dtype=np.int16)
    depth = int(difficulty / VERTICAL_SCALE)
    platform_half = int(4.0 / HORIZONTAL_SCALE / 2)
    center = count // 2
    field[center - platform_half : center + platform_half, center - platform_half : center + platform_half] = -depth
    return field


def _height_field(name: str, difficulty: float, rng: np.random.Generator) -> np.ndarray:
    if name == "flat":
        return _pyramid_slope(difficulty, 1.0, False, rng) * 0
    if name == "smooth_slope_positive":
        return _pyramid_slope(difficulty, 1.0, False, rng)
    if name == "smooth_slope_negative":
        return _pyramid_slope(difficulty, -1.0, False, rng)
    if name == "rough_slope_positive":
        return _pyramid_slope(difficulty, 1.0, True, rng)
    if name == "rough_slope_negative":
        return _pyramid_slope(difficulty, -1.0, True, rng)
    if name == "stairs_negative":
        return _stairs(difficulty, -1.0)
    if name == "stairs_positive":
        return _stairs(difficulty, 1.0)
    if name == "discrete_obstacles":
        return _discrete_obstacles(difficulty, rng)
    if name == "stepping_stones":
        return _stepping_stones(difficulty, rng)
    if name == "gap":
        return _gap(difficulty)
    if name == "pit":
        return _pit(difficulty)
    raise KeyError(name)


def fudan_sub_terrain(difficulty: float, cfg: "FudanSubTerrainCfg"):
    """Generate one collision-ready tile from the reference height formulas."""
    rng = np.random.default_rng(cfg.tile_seed)
    field = _height_field(cfg.variant, float(difficulty), rng)
    # A standalone tile needs the shared boundary sample that the reference's
    # monolithic height field obtains from its neighbouring tile.
    field = np.pad(field, ((0, 1), (0, 1)), mode="edge")
    vertices, faces = convert_height_field_to_mesh(
        field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    middle = field.shape[0] // 2
    radius = int(1.0 / cfg.horizontal_scale)
    center_patch = field[middle - radius : middle + radius, middle - radius : middle + radius]
    origin_z = float(np.max(center_patch) * cfg.vertical_scale)
    return [mesh], np.asarray((cfg.size[0] * 0.5, cfg.size[1] * 0.5, origin_z))


@configclass
class FudanSubTerrainCfg(HfTerrainBaseCfg):
    """Configuration for one exact Fudan terrain branch."""

    function = fudan_sub_terrain
    variant: str = "flat"
    tile_seed: int = 1


class FudanTerrainGenerator(TerrainGenerator):
    """Terrain generator with exact ``difficulty = row / 10`` levels."""

    def _generate_curriculum_terrains(self):
        proportions = np.asarray([cfg.proportion for cfg in self.cfg.sub_terrains.values()], dtype=np.float64)
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)
        sub_cfgs = list(self.cfg.sub_terrains.values())
        base_seed = int(self.cfg.seed or 0)
        for col in range(self.cfg.num_cols):
            sub_index = int(np.min(np.where(col / self.cfg.num_cols + 0.001 < cumulative)[0]))
            for row in range(self.cfg.num_rows):
                difficulty = row / self.cfg.num_rows
                cfg = sub_cfgs[sub_index].copy()
                cfg.tile_seed = base_seed + row * 1009 + col * 9176
                mesh, origin = self._get_terrain_mesh(difficulty, cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)


FUDAN_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=FudanTerrainGenerator,
    seed=1,
    curriculum=True,
    size=(TILE_SIZE, TILE_SIZE),
    border_width=25.0,
    num_rows=NUM_LEVELS,
    num_cols=20,
    horizontal_scale=HORIZONTAL_SCALE,
    vertical_scale=VERTICAL_SCALE,
    slope_threshold=SLOPE_THRESHOLD,
    difficulty_range=(0.0, 0.9),
    # Per-vertex height coloring roughly doubles the peak memory required by
    # this 200-tile training mesh.  A single preview material is sufficient to
    # read the terrain shape, while collision geometry remains unchanged.
    color_scheme="none",
    use_cache=False,
    sub_terrains={
        name: FudanSubTerrainCfg(proportion=proportion, variant=name)
        for name, proportion in zip(TRAINING_VARIANTS, TRAINING_PROPORTIONS)
    },
)


def build_fudan_terrain_grid(seed: int = 1) -> FudanTerrainGrid:
    """Build the deterministic 10-level by 11-visible-variant preview mesh."""
    meshes: list[trimesh.Trimesh] = []
    origins = np.zeros((NUM_LEVELS, len(COLUMN_NAMES), 3), dtype=np.float64)
    difficulties = np.arange(NUM_LEVELS, dtype=np.float64) / NUM_LEVELS
    palette = trimesh.visual.color.interpolate(
        np.linspace(0.05, 0.95, len(COLUMN_NAMES)), color_map="turbo"
    )
    map_x = NUM_LEVELS * TILE_SIZE
    map_y = len(COLUMN_NAMES) * TILE_SIZE
    for row, difficulty in enumerate(difficulties):
        for col, name in enumerate(COLUMN_NAMES):
            rng = np.random.default_rng(seed + row * 1009 + col * 9176)
            field = _height_field(name, float(difficulty), rng)
            vertices, faces = convert_height_field_to_mesh(
                field, HORIZONTAL_SCALE, VERTICAL_SCALE, SLOPE_THRESHOLD
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            tile_translation = np.array(
                (row * TILE_SIZE - map_x / 2, col * TILE_SIZE - map_y / 2, 0.0)
            )
            mesh.apply_translation(tile_translation)
            color = np.asarray(palette[col], dtype=np.uint8)
            mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            meshes.append(mesh)
            middle = field.shape[0] // 2
            radius = int(1.0 / HORIZONTAL_SCALE)
            center_patch = field[middle - radius : middle + radius, middle - radius : middle + radius]
            origin_z = np.max(center_patch) * VERTICAL_SCALE
            origins[row, col] = tile_translation + (TILE_SIZE / 2, TILE_SIZE / 2, origin_z)
    return FudanTerrainGrid(
        mesh=trimesh.util.concatenate(meshes),
        origins=origins,
        difficulties=difficulties,
        column_names=COLUMN_NAMES,
    )
