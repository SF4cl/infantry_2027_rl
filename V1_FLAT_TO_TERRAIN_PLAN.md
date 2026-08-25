# v1 平地到地形训练实施记录

更新时间：2026-08-25

## 目标与版本边界

本路线的目标是从随机网络开始，先在平地得到一个完整可用的策略，再无结构变化地续训到 Fudan 最终实际地形集合，之后进入 MuJoCo sim2sim。旧的 `Infantry-2027-*-v0`、已有日志和 checkpoint 保持不变；本轮新增任务全部使用 `v1` 名称。

阶段如下：

1. `Infantry-2027-Flat-Compatible-v1`：4096 环境，从零训练到 2000 updates。
2. `Infantry-2027-Terrain-v1`：完整加载平地 checkpoint（Actor、Critic、Encoder、两个 optimizer 和 iteration），从最低地形等级开始续训。
3. 固定地形可视化与正/倒车分桶验收。
4. 导出最新策略，在不可变 `infantry_2027_v0` MuJoCo 闭链资产上做 sim2sim。

## 中立摆角检查

未采用“总质心与轮轴水平投影”直接换算出的约 `-0.04 rad` 粗略结果。两轮机器人是欠驱动倒立系统，没有稳定外环时的自由落体扫描会把控制器失稳误当作 VMC 平衡点。

新增 `mujoco_sim2sim/scripts/calibrate_equilibrium_angle.py`，使用独立的水平/姿态软测量架约束欠驱动自由度，竖直方向保持自由，扫描目标腿长和 VMC 摆角，并以所需保持力、保持力矩、速度和姿态误差综合判断。3 秒扫描结果为：

| 目标腿长 | 动态目标摆角 |
|---:|---:|
| 0.16 m | 0.000 rad |
| 0.22 m | 0.000 rad |
| 0.28 m | -0.005 rad |
| 0.33 m | -0.005 rad |

报告保存在 `mujoco_sim2sim/results/equilibrium_angle_mujoco.json`。v1 执行层按腿长线性插值以上微小偏置；v0 默认表为空，因此旧 checkpoint 的动作语义没有变化。训练中新增正向、反向和静止三组速度误差累计量与采样占比，实际方向 MAE 应使用 `error_sum / fraction` 计算，避免掩码外样本稀释结果。

## 平地与地形 checkpoint 兼容性

两阶段使用相同网络输入：

- Actor：5 帧 × 25 维 proprioception，共 125 维。
- Encoder：`125 -> [128, 64] -> 3`，监督目标是真实机体系三维线速度乘观察尺度 2.0。
- Actor MLP：当前帧 25 维与 Encoder 3 维拼接，输入 28 维，隐藏层 `[128, 64, 32]`，输出 6 维。
- Critic：145 维，隐藏层 `[256, 128, 64]`。
- 145 维 Critic 组成：3 + 25 + 6 + 6 + 6 + 77 + 6 + 16。

平地也使用同一个 11×7 RayCaster。其地形由本地 1×1、难度 0、纯 flat terrain generator 创建，不依赖在线 GroundPlane USD；77 维高度形貌在平地为平坦信息，但维数和地形阶段完全一致。

已完成验证：

- 平地 8 环境、2 updates 的 PPO/Encoder 烟雾测试通过，生成 `model_2.pt`。
- 地形 8 环境、1 update 的烟雾测试通过。
- 两个 checkpoint 的 23 个模型参数键和每个张量形状完全一致。
- 使用新增 `--resume_path` 将平地 `model_2.pt` 完整续训到地形，总目标 3 updates 时正确报告 `remaining updates: 1`，并生成地形 `model_3.pt`。

## 最终 Fudan 奖励对齐

v1 平地和地形使用参考最终目录 `6010.2加入随机地形terrain_proportions0.50.20.10.10.10.0` 中实际启用的同一集合：

| 奖励项 | 权重 | 说明 |
|---|---:|---|
| tracking_lin_vel | 1.0 | 函数内部乘 1.3 |
| tracking_lin_vel_enhance | 1.0 | 函数内部乘 1.45 |
| tracking_ang_vel | 1.0 | 不启用 yaw enhance |
| base_height | 1.0 | 函数内部乘 1.5；使用局部地面高度 |
| nominal_state | -0.1 | 只约束左右腿摆角一致 |
| lin_vel_z | -0.1 |  |
| ang_vel_xy | -0.05 |  |
| orientation | -10.0 |  |
| dof_vel | -1e-6 |  |
| dof_acc | -1e-8 |  |
| torques | -1e-5 |  |
| action_rate | -0.003 |  |
| action_smooth | -0.003 |  |
| collision | -1.0 | base、lf、rf |
| dof_pos_limits | -1.0 | CAD 碰撞确定的左右前膝相对限位 |

没有启用参考最终配置中注释掉的 `tracking_ang_vel_enhance`、`base_height_enhance`、`wheel_air_theta0`，也没有加入旧仓库的停止位移、滑移、饱和、倾角 barrier 等额外奖励。

PPO 采用参考参数：48 steps/env、5 epochs、4 minibatches、adaptive learning rate 1e-3、gamma 0.99、lambda 0.95、desired KL 0.005、entropy 0.01、从零平地初始 action noise std 0.5。该数值以参考仓库实际入口 `plane/wheel_legged_gym/envs/base/legged_robot_config.py::LeggedRobotCfgPPO.policy` 为准；部分后期 terrain/recovery 日志快照中的 1.0 不代表从零平地入口。

2026-08-25 的首轮 v1 平地曾误用后期日志快照的 `init_noise_std=1.0`。训练到 update 500 时，六维 std 从初始 1.0 演化为 `[0.717, 1.520, 2.787, 0.679, 1.603, 2.666]`，最近 20 轮 episode length 从 update 200 的 1751 降到 1423，确认高探索噪声持续损害随机 rollout。该实验完整保留为诊断对照，正式 v1 从零训练改回参考入口的 0.5 后重新开始。

## DR 与命令

完整最终 DR 从平地第 0 次更新即一次性启用，进入地形时不再切换动力学分布：摩擦、恢复系数、base mass/inertia/COM、默认关节位置、VMC Kp/Kd、电机强度、0～10 ms 动作延迟，以及每 7 秒的水平推扰或向下冲击。

平地保留完整最终命令能力：`vx ±2.3 m/s`、运动转向最终 `±4 rad/s`、原地转最终 `±10 rad/s`、base height `0.148～0.318 m`（腿长 `0.16～0.33 m`）。命令每 10 秒瞬时重采样；前 1500 updates 连续扩展范围。

地形仍保留倒车穿越：

- flat 列保持完整 direct yaw、原地转和运动转向分布。
- 非平地列以 50/50 采样正向/倒向 body-x 速度，最小绝对速度 0.20 m/s。
- 非平地列不采样横着冲向障碍的持续转向命令，而使用 `1.5 × heading error` 的 yaw-rate 修正，限幅 `±1 rad/s`，将机体纵轴保持在地形网格 x 轴附近。
- 这不是禁止倒车：负 `vx` 时机体仍保持同一朝向，以车尾方向跨越地形。

地形集合严格为参考最终实际训练集合：flat 50%，正/负平滑坡各 10%，正/负粗糙坡各 5%，下/上楼梯各 10%。平地成熟 checkpoint 进入地形时从 level 0 开始，而不是照搬参考仓库面向成熟 checkpoint 的 `max_init_terrain_level=5`。

## 命令

正式平地从零训练：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Flat-Compatible-v1 --num_envs 4096 --max_iterations 2000 --headless --run_name flat_compatible_v1_4096x2000
```

平地中断后续训到总计 2000 updates：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Flat-Compatible-v1 --num_envs 4096 --max_iterations 2000 --headless --resume --load_run <run目录名> --checkpoint model_<轮数>.pt
```

平地完成后完整续训到地形：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Terrain-v1 --num_envs 1024 --max_iterations 5000 --headless --run_name terrain_v1_from_flat --resume_path <平地model_2000.pt绝对路径>
```

这里 `5000` 是总 update 目标。若载入的是 `model_2000.pt`，脚本只会再训练 3000 updates。

TensorBoard：

```powershell
D:\condaenvs\isaacsim510\python.exe -m tensorboard.main --logdir D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext\logs\rsl_rl\infantry_2027_v1_flat_compatible --port 6006
```

平地可视化：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\play.py --task Infantry-2027-Flat-Compatible-Play-v1 --num_envs 1 --checkpoint <model.pt> --real-time --keyboard
```

## 阶段验收

平地 2000 updates 完成后不会只凭总 reward 决定是否进地形。至少检查：`non_finite=0`、episode length、姿态/碰撞项无退化、Encoder 三轴 RMSE、base height、yaw、正向 MAE、反向 MAE、静止漂移，以及最终范围的键盘可视化。正反向 MAE 明显不对称时，先检查分桶指标和策略动作均值，不再凭肉眼修改 VMC 几何符号。

地形阶段检查 level 分布、正/倒向成功率、是否出现侧向接近、不同地形列的碰撞与超时原因。只有 Isaac Sim 固定地形验收通过后，才导出该 checkpoint 进入 MuJoCo sim2sim。
