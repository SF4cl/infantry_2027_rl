"""Preview all terrain scenes from ``ref/fudan_rl_wheel_leg/plane`` in Isaac Sim.

The signed slope and rough-slope variants are placed in separate columns, so
the original nine terrain branches become eleven visible columns.
"""

import argparse

# Isaac Sim's GUI extensions bundle an HDF5 runtime.  On Windows, importing
# h5py only after Kit starts can bind h5py._errors to that incompatible DLL
# and terminate Python with 0xc0000139.  Load the environment's native
# extensions before AppLauncher changes the DLL search/load state.  Keep this
# in sync with scripts/rsl_rl/play.py.
import h5py  # noqa: F401, E402
import torch  # noqa: F401, E402

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the complete Fudan plane terrain set.")
parser.add_argument("--seed", type=int, default=1, help="Deterministic seed for rough and obstacle terrains.")
parser.add_argument(
    "--validation_steps",
    type=int,
    default=4,
    help="Number of physics steps before exiting in headless mode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from pxr import UsdGeom

import isaaclab.sim as sim_utils

from infantry_2027.terrains import build_fudan_terrain_grid


def create_visual_mesh(prim_path, mesh):
    """Create the full-resolution preview mesh without expensive collision cooking."""
    sim_utils.create_prim(prim_path, "Xform")
    prim = sim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.full(len(mesh.faces), 3, dtype=np.int32),
            "subdivisionScheme": "none",
        },
    )
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32) / 255.0
    color_attr = prim.GetAttribute("primvars:displayColor")
    UsdGeom.Primvar(color_attr).SetInterpolation(UsdGeom.Tokens.vertex)
    color_attr.Set(colors[:, :3])


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[72.0, 74.0, 67.0], target=[0.0, 0.0, -0.5])

    light_cfg = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.82, 0.82, 0.82))
    light_cfg.func("/World/Light", light_cfg)

    grid = build_fudan_terrain_grid(seed=args_cli.seed)
    create_visual_mesh("/World/FudanTerrainGrid", grid.mesh)

    sim.reset()
    bounds = np.asarray(grid.mesh.bounds)
    print("\n[FUDAN TERRAIN PREVIEW]")
    print("  X axis: difficulty rows, d = 0.0 ... 0.9")
    print("  Y axis: terrain columns")
    for index, name in enumerate(grid.column_names):
        print(f"    {index:2d}: {name}")
    print(
        f"  tiles={grid.origins.shape[0] * grid.origins.shape[1]}, "
        f"vertices={len(grid.mesh.vertices)}, faces={len(grid.mesh.faces)}"
    )
    print(f"  bounds_min={bounds[0].round(3).tolist()}, bounds_max={bounds[1].round(3).tolist()}")

    if args_cli.headless:
        for _ in range(max(1, args_cli.validation_steps)):
            sim.step()
        print("FUDAN_TERRAIN_PREVIEW_OK")
    else:
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
