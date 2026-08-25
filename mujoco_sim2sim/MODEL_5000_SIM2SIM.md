# model_5000 MuJoCo sim2sim

Isaac Lab run:

```text
2026-08-24_00-53-14_long_v1_direct_yaw/model_5000.pt
checkpoint iteration: 5000
completed iterations: 5000
checkpoint SHA-256: A00F1FE8FEA8965740FDAA9ECDC7BDB4621C6349110B5553078E398A05969E9F
```

Exported policy:

```text
mujoco_sim2sim/exported/model_5000.npz
schema: infantry-2027-v0-fudan-estimator
contract: infantry-2027-v0-flat-25d-v1
```

The old `model_1600.npz` export and its validation files remain unchanged.

## Nominal-PD validation

The first pass used the exact training nominal VMC gains:

```text
length Kp/Kd = 900 / 20
angle  Kp/Kd = 50 / 3
```

All constant-command tests survived without numerical divergence or falling.

| Command | Measured result | P90 tilt | Max closure residual |
|---|---:|---:|---:|
| stand | vx = -0.114 m/s | 0.92 deg | 0.295 mm |
| vx = 0.8 m/s | vx = 0.604 m/s | 0.95 deg | 0.295 mm |
| vx = -0.8 m/s | vx = -0.866 m/s | 0.79 deg | 0.295 mm |
| vx/yaw = 0.8/0.8 | vx/yaw = 0.607/0.796 | 1.19 deg | 0.354 mm |
| vx = 2.3 m/s | vx = 1.550 m/s | 2.54 deg | 0.295 mm |
| vx = -2.3 m/s | vx = -2.107 m/s | 1.86 deg | 0.295 mm |
| yaw = 3 rad/s | yaw = 2.993 rad/s | 1.53 deg | 0.388 mm |
| vx/yaw = 0.8/3.0 | vx/yaw = 0.406/2.912 | 1.25 deg | 0.410 mm |
| minimum height 0.148 m | actual = 0.154 m | 0.54 deg | 0.336 mm |
| maximum height 0.318 m | actual = 0.301 m | 0.85 deg | 0.407 mm |

The combined height-step diagnostic exposed a transient hidden by constant
command averages: a direct minimum-to-maximum height jump makes the nominal-PD
base height overshoot to about 0.39 m before settling.  The policy remains
upright, but this response is underdamped and should not be accepted as the
final sim2sim controller.

## Rechecked PD candidates on model_5000

The moderate candidate was re-evaluated rather than copied blindly from the
old policy:

```text
length Kp/Kd = 1200 / 100
angle  Kp/Kd = 80 / 6
```

It reduces height-jump peak to about 0.328 m, has no measured 45 Nm saturation,
and changes the 20 s diagnostic metrics as follows:

| Metric | Nominal PD | Moderate PD |
|---|---:|---:|
| velocity MAE | 0.193 m/s | 0.171 m/s |
| length tracking RMSE | 18.93 mm | 14.35 mm |
| angle tracking RMSE | 0.0554 rad | 0.0438 rad |
| peak inner leg effort | 36.93 Nm | 39.42 Nm |
| P90 tilt | 1.17 deg | 1.42 deg |

At the final envelope it reaches 1.603 m/s for a 2.3 m/s forward command,
-2.052 m/s for -2.3 m/s, and 0.539 m/s plus 2.961 rad/s for a 0.8 m/s +
3 rad/s command.  It improves forward/high-yaw tracking, while backward
tracking is slightly worse than nominal.

The more aggressive `1800/180, 120/8` group tracks more tightly but touches the
45 Nm limit during height jumps and has higher tilt.  It is retained only as a
comparison, not the recommended viewer default.

## Viewer command

Run from `D:\rm\2026_code\rl\infantry_2027_rl`.

Recommended moderate PD:

```powershell
python .\mujoco_sim2sim\scripts\run_policy.py `
  --policy .\mujoco_sim2sim\exported\model_5000.npz `
  --viewer --keyboard --duration 600 `
  --kp-length 1200 --kd-length 100 --kp-angle 80 --kd-angle 6 `
  --report-interval 0.25
```

Exact training nominal PD for comparison:

```powershell
python .\mujoco_sim2sim\scripts\run_policy.py `
  --policy .\mujoco_sim2sim\exported\model_5000.npz `
  --viewer --keyboard --duration 600 `
  --kp-length 900 --kd-length 20 --kp-angle 50 --kd-angle 3 `
  --report-interval 0.25
```

The terminal reports policy target and measured leg length/angle plus the 500 Hz
inner-loop peak effort.  The complete command/target/response plots are stored
under `mujoco_sim2sim/results/model5000/`.
