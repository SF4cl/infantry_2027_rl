# infantry_2027_rl server migration manifest

Prepared on 2026-08-25 for resuming the formal v1 training chain on an Ubuntu server.

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
- Initial environment counts must remain unchanged: flat `4096`, terrain `1024`

## Resume checkpoint

- Relative path: `isaaclab_ext/logs/rsl_rl/infantry_2027_v1_flat_compatible/2026-08-25_04-14-29_flat_compatible_v1_4096x2000_std05_restart/model_850.pt`
- SHA-256: `AA2C0DDE524647F710EBEC3BFBC047B3098F5FA881C6309258BA8AF60141F4C2`
- Stored `completed_iterations`: `851`
- State includes Actor/Critic/Encoder weights, PPO optimizer state, and estimator optimizer state.

The checkpoint is from the same from-scratch formal run. Resuming it on another GPU continues that training chain; it does not turn the experiment into pretrained-policy initialization.

## Required layout

Extract the archive below the server data disk so the resulting layout is:

```text
/root/autodl-tmp/infantry_2027_rl/
  assets/infantry_2027_v0/
  isaaclab_ext/
  mujoco_sim2sim/
  README.md
  TRAINING_DESIGN.md
  V1_FLAT_TO_TERRAIN_PLAN.md
```

Do not move `assets/infantry_2027_v0` inside `isaaclab_ext`; the asset loader intentionally resolves it from the project root.

## First server checks

```bash
cd /root/autodl-tmp/infantry_2027_rl/isaaclab_ext
python -m pip install -e source/infantry_2027
python -c "import isaacsim, isaaclab; print(isaaclab.__version__)"
python scripts/list_envs.py
sha256sum logs/rsl_rl/infantry_2027_v1_flat_compatible/2026-08-25_04-14-29_flat_compatible_v1_4096x2000_std05_restart/model_850.pt
```

The hash printed by `sha256sum` must match this manifest before training resumes. Detailed launch and automatic flat-to-terrain monitor commands are recorded in `isaaclab_ext/README.md`.
