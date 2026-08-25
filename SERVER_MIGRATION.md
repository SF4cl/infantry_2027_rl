# infantry_2027_rl server runtime contract

This document records only the environment and synchronization contract needed to reproduce training.
Checkpoints, logs, metric reports, and one-off audit scripts are transferred as local artifacts and are not tracked by Git.

## Runtime contract

- Isaac Sim: `5.1.0`
- Isaac Lab: `v2.3.2`
- Isaac Lab commit used locally: `37ddf626871758333d6ed89cf64ad702aef127d0`
- Python: `3.11`
- PyTorch: `2.7.0+cu128`
- RSL-RL: `3.1.2`
- Gymnasium: `1.2.1`
- TensorBoard: `2.20.0`
- NumPy: `1.26.0`
- Target GPU: one RTX 4090 24GB
- Formal flat run: `4096` environments and `2000` updates from random network weights.
- Terrain environment count is selected separately after flat-policy review.

## Required layout

The server working copy must have this layout:

```text
/root/gpufree-data/rl/infantry_2027_rl/
  assets/infantry_2027_v0/
  isaaclab_ext/
  mujoco_sim2sim/
  README.md
  TRAINING_DESIGN.md
```

Do not move `assets/infantry_2027_v0` inside `isaaclab_ext`; the asset loader intentionally resolves it from the project root.

## First server checks

```bash
cd /root/gpufree-data/rl/infantry_2027_rl
git pull --ff-only origin main
python -m pip install -e isaaclab_ext/source/infantry_2027
cd isaaclab_ext
python -c "import isaacsim, isaaclab; print(isaaclab.__version__)"
python scripts/list_envs.py --keyword Infantry
bash scripts/automation/start_flat_scratch_server.sh
```

Run the launcher inside `tmux`. It executes `train.py` in the foreground and does not automatically start terrain training.
Do not pull or reinstall the working copy while a training process is running.
