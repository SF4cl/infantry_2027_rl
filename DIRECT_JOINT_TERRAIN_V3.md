# Direct-joint mixed-terrain v3

`Infantry-2027-Joint-Terrain-v3` 是新的正式从零训练入口。旧的 v0、v1、v2
任务、VMC 实现和 checkpoint 均保留；v3 checkpoint 与旧版本不兼容，禁止续训旧模型。

## 能力范围

- 从接近站立的已验证初始姿态开始；暂不训练倒地起身或 recovery。
- 单次训练同时包含平地、正负坡、正负粗糙坡和上下楼梯。
- 最终前进/后退命令为 `[-2.5, 2.5] m/s`。
- 保留原地转向；运动 yaw-rate 最终上限为 `4 rad/s`，原地为 `10 rad/s`。
- base-height 命令范围为 `0.148～0.318 m`，对应名义腿长 `0.16～0.33 m`。

## 六维直接电机动作

策略顺序严格采用参考实现：

```text
[left_front, left_rear, left_wheel, right_front, right_rear, right_wheel]
```

腿电机目标为：

```text
q_target = randomized_default_q + 0.5 * action
tau = 60 * Kp_scale * (q_target - q) - 1 * Kd_scale * qd
```

轮电机目标为：

```text
qd_target = 10 * action
tau = 0.2 * Kd_scale * (qd_target - qd)
```

腿、轮力矩分别限制为 `45 Nm` 和 `5 Nm`。PD、输出力矩缩放均在 `0.9～1.1`
随机化，动作延迟为 `0～10 ms`。控制链不再计算 VMC 力或腿长 residual；五连杆正运动学
只用于计算参考仓库已有的左右腿摆角差奖励。

## 训练分布

20 个地形列的最终比例为：

| 地形 | 比例 |
|---|---:|
| 平地 | 50% |
| 正坡 / 负坡 | 各 10% |
| 正粗糙坡 / 负粗糙坡 | 各 5% |
| 下楼梯 / 上楼梯 | 各 10% |

所有环境从难度 level 0 开始。单个环境行进距离超过 `4 m` 升级；未升级且该 episode
线速度跟踪得分低于 `0.4` 时降级。非平地允许正向和倒向穿越，但 heading 控制会让机体
保持大致平行于地形行进轴，避免机械上不合理的侧向跨越。

速度、转向和高度只使用同一个 run 内的命令课程，不拆分 checkpoint 阶段。前 1500 个
PPO updates 中，`vx` 从 `±0.5` 连续扩展到 `±2.5 m/s`；运动 yaw 从 `±1` 扩展到
`±4 rad/s`，原地 yaw 从 `±2` 扩展到 `±10 rad/s`，高度从 `0.188～0.248 m`
扩展到 `0.148～0.318 m`。每 5 秒瞬时重采样一次命令。

## 观测、奖励和 PPO

- Actor：Fudan 25 维本体观测的五帧历史，共 125 维。
- Encoder：`125 -> [128, 64] -> 3`，监督三维机体系线速度。
- Actor 实际输入：当前 25 维加 Encoder 3 维，共 28 维。
- Critic：145 维 privileged observation，包含 77 维地形扫描。
- Reward：严格使用参考最终 `6010.2` 快照中的 15 项奖励和权重；不加入旧仓库额外奖励。
- PPO：48 steps/env、5 epochs、4 mini-batches、学习率 `1e-3 adaptive`、
  `gamma=0.99`、`lambda=0.95`、`KL=0.005`、entropy `0.01`。
- 默认长训练：4096 环境、10000 updates、每 20 updates 保存 checkpoint。

完整 DR 从第 0 次更新启用：摩擦 `0.1～2.0`、恢复系数 `0～1`、base mass
`-2～+3 kg`、惯量 `0.8～1.2`、COM 三轴 `±0.05 m`、默认关节角 `±0.05 rad`，
并每 7 秒施加水平推扰或向下冲击。4096 环境已经用于此前服务器地形训练；不要改为
8192，否则可能再次触发 PhysX collision stack overflow。

## 本机短测试

先用独立 run 做 200 updates 验证直接关节版本的学习趋势：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Joint-Terrain-v3 --num_envs 128 --max_iterations 200 --headless --run_name joint_terrain_v3_128x200_gate
```

## 服务器同步和从零长训练

本机完成审核和提交后推送：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl
git push origin main
git push server main
```

服务器在没有旧训练进程占用仓库时同步：

```bash
cd /root/gpufree-data/rl/infantry_2027_rl
git status --short
git pull --ff-only
python -m pip install -e isaaclab_ext/source/infantry_2027
git log -1 --oneline
```

在 tmux 前台从随机网络权重开始长训练：

```bash
tmux new -s joint_terrain_v3
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
python -u scripts/rsl_rl/train.py \
  --task Infantry-2027-Joint-Terrain-v3 \
  --num_envs 4096 \
  --max_iterations 10000 \
  --headless \
  --run_name joint_terrain_v3_4096x10000_server_scratch
```

这里故意没有 `--resume` 或 `--resume_path`。按 `Ctrl-b` 后按 `d` 脱离 tmux；
`tmux attach -t joint_terrain_v3` 返回训练终端。正常暂停使用 `Ctrl+C`，训练脚本会保存
`model_interrupted_<iteration>.pt`。

## 可视化

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
python scripts/rsl_rl/play.py \
  --task Infantry-2027-Joint-Terrain-Play-v3 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model_xxx.pt \
  --terrain-type stairs_up \
  --terrain-level 3 \
  --real-time \
  --keyboard
```

键盘保持 `I/K` 前后、`J/L` 转向、`U/O` 高度、`M` 停止、`N` 恢复默认高度。
v3 的默认键盘前后速度为 `±2.5 m/s`；旧 v0～v2 任务仍会自动限制在各自的
`±2.3 m/s` 训练范围内。
