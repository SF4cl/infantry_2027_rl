# Terrain-Stable-v2 training

`Infantry-2027-Terrain-Stable-v2` is the reviewed continuation stage for the
accepted `Infantry-2027-Flat-Stable-v2` checkpoint. It is not intended to start
from random weights.

## Runtime contract

- actor observations: 5 x 25 = 125
- critic observations: 145
- estimator target: 3-D body linear velocity
- action: verified 6-D VMC/wheel action
- environments: 1024
- initial terrain level: 0
- terrain curriculum levels: 0 through 9
- command duration: 5 seconds
- final forward range: -2.3 through 2.3 m/s
- final base-height range: 0.148 through 0.318 m
- training checkpoint interval: 20 updates

The task preserves the accepted Flat-Stable-v2 rewards and moderate domain
randomization. Terrain-specific changes are limited to local-ground base-height
measurement, the terrain curriculum, aligned forward/reverse traversal commands,
and contact penalties on `base_link`, `lf_link`, and `rf_link` only.

## Server synchronization

Push the reviewed commit from the Windows working copy:

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl
git push origin main
git push server main
```

Update the server working copy only while no training process is using it:

```bash
cd /root/gpufree-data/rl/infantry_2027_rl
git status --short
git pull --ff-only
python -m pip install -e isaaclab_ext/source/infantry_2027
git log -1 --oneline
```

## Locate the accepted flat checkpoint

Run from the external-project directory:

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
FLAT_CKPT=$(find logs/rsl_rl/infantry_2027_v2_flat_stable -type f -name 'model_2000.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
test -n "$FLAT_CKPT" && test -f "$FLAT_CKPT"
echo "$FLAT_CKPT"
```

If the checkpoint was stored outside `logs`, set `FLAT_CKPT` to that absolute
path instead.

## Recommended 200-update gate

The checkpoint records the next learning iteration as 2001. An absolute target
of 2200 therefore performs 199 additional updates. Start it in a tmux session:

```bash
tmux new -s terrain_v2
```

Then run in the tmux foreground:

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
python -u scripts/rsl_rl/train.py \
  --task Infantry-2027-Terrain-Stable-v2 \
  --num_envs 1024 \
  --max_iterations 2200 \
  --headless \
  --resume_path "$FLAT_CKPT" \
  --run_name terrain_stable_v2_1024x2200_server_gate
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t terrain_v2
```

## Long terrain continuation

After accepting the gate checkpoint, locate it and continue to the absolute
target iteration 5000:

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
TERRAIN_CKPT=$(find logs/rsl_rl/infantry_2027_v2_terrain_stable -type f -name 'model_2200.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
test -n "$TERRAIN_CKPT" && test -f "$TERRAIN_CKPT"
python -u scripts/rsl_rl/train.py \
  --task Infantry-2027-Terrain-Stable-v2 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --headless \
  --resume_path "$TERRAIN_CKPT" \
  --run_name terrain_stable_v2_1024x5000_server_long
```

To skip the gate and directly run the full terrain continuation, use the same
long command with `--resume_path "$FLAT_CKPT"`. The target 5000 is absolute,
so a run resumed from the accepted flat checkpoint performs about 3000 terrain
updates rather than 5000 additional updates.

Pressing `Ctrl+C` invokes the safe interruption path and writes
`model_interrupted_<iteration>.pt`. Do not use `kill -9` for a normal stop.
