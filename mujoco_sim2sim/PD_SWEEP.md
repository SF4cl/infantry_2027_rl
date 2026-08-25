# model_1600 MuJoCo VMC PD sweep

This experiment keeps the exported `model_1600` policy, observation history,
500 Hz MuJoCo physics, 100 Hz policy, closed-chain asset, contacts, torque
limits and wheel controller unchanged.  Only the four explicit VMC gains are
changed.

## Coverage

- Coarse grid: 120 gain groups.
- Boundary refinement: 80 gain groups.
- Total: 200 gain groups.
- Every group was tested on stand, forward, backward, forward+turn, minimum
  height, maximum height, height steps and motion/stop steps.
- Every group survived all eight test profiles.
- The sweep records the policy's left/right target leg length and angle, the
  measured VMC state, raw action, 500 Hz inner-loop peak effort, effort
  saturation, tilt, velocity tracking and closure residual.

Machine-readable results and plots are in `results/policy_pd_sweep/`:

- `refined_summary.json`: all metrics for all 200 groups.
- `refined_ranking.csv`: compact combined ranking.
- `score_heatmap.png` and `refinement_heatmap.png`: coarse/refined score maps.
- `baseline_diagnostic.png`: original `900/20, 50/3` response.
- `refined_best_diagnostic.png`: numerical best `1800/180, 120/8` response.

## Main result

The original training gains are:

```text
length Kp/Kd = 900 / 20
angle  Kp/Kd = 50 / 3
```

They rank 155/200 under the MuJoCo-only composite score.  The numerical best is:

```text
length Kp/Kd = 1800 / 180
angle  Kp/Kd = 120 / 8
```

Compared with the baseline, its aggregate target-to-actual length RMSE changes
from 13.91 mm to 8.20 mm, angle RMSE from 0.0485 rad to 0.0282 rad, and velocity
MAE from 0.199 m/s to 0.145 m/s.  Mean inner-loop saturation is 0.023%; height
steps briefly hit the 45 Nm limit.  This is a MuJoCo numerical optimum, not a
new training or hardware default: it is well outside the policy's +/-5% gain
randomization range.

A less aggressive comparison point is:

```text
length Kp/Kd = 1200 / 100
angle  Kp/Kd = 80 / 6
```

It improves tracking substantially while moving less far from the training
controller.  Neither candidate should replace the Isaac Lab training gains
without an Isaac-side test and, for a final policy, retraining or much wider
gain randomization.

## Visual comparison

Run from the repository root.  All commands print the policy target leg length
and the measured leg length every 0.25 s.

Baseline:

```powershell
python .\mujoco_sim2sim\scripts\run_policy.py --viewer --keyboard --duration 600 `
  --kp-length 900 --kd-length 20 --kp-angle 50 --kd-angle 3 `
  --report-interval 0.25
```

Moderate candidate:

```powershell
python .\mujoco_sim2sim\scripts\run_policy.py --viewer --keyboard --duration 600 `
  --kp-length 1200 --kd-length 100 --kp-angle 80 --kd-angle 6 `
  --report-interval 0.25
```

MuJoCo numerical best:

```powershell
python .\mujoco_sim2sim\scripts\run_policy.py --viewer --keyboard --duration 600 `
  --kp-length 1800 --kd-length 180 --kp-angle 120 --kd-angle 8 `
  --report-interval 0.25
```

The policy target length is decoded as:

```text
target_length = clip(base_height + 0.012 + tanh(length_action) * 0.03,
                     0.16, 0.33)
```

Thus the base-height command is a nominal length, not the final motor-level
target.  During straight motion both targets stay close together; during a
turn the policy deliberately separates left and right target lengths.  At the
minimum height negative residuals are clipped at 0.16 m.  At the maximum height
the policy can still reduce the target below 0.33 m with its residual action.
