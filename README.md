# infantry_2027_rl

基于高保真 `infantry_2027_v0` 闭链资产的 Isaac Lab 训练与 MuJoCo sim2sim 根目录。

## 目录

```text
infantry_2027_rl/
├─ assets/infantry_2027_v0/   # 只读、不可变资产快照
├─ isaaclab_ext/              # Isaac Lab 官方模板生成的外部仓库
└─ mujoco_sim2sim/            # 后续 policy sim2sim；不复制另一份资产
```

资产快照共 333 个文件、848256141 bytes。训练和 sim2sim 都必须引用这份快照，禁止原地重新转换或修改；若资产发生变化，应建立 `infantry_2027_v1`。

## 当前训练契约

- 物理：500 Hz（`dt=0.002`），策略 100 Hz（decimation 5），20 s episode。
- 地板由本地 USD 几何直接生成，启动训练不依赖 NVIDIA S3/Nucleus 网络资源。
- 动作：`[left_angle, left_length_residual, left_wheel, right_angle, right_length_residual, right_wheel]`。
- VMC：`l1=0.215 m`、`l2=0.2537 m`，单腿支撑前馈 118.88 N，腿电机 45 Nm，轮电机 5 Nm。
- 指令：`vx + direct yaw-rate + base-height`；10% 站立、20% 原地转向、70% 运动分布。
- 最终范围：`vx ±2.3 m/s`，运动 yaw-rate 上限 `±4 rad/s`，原地 yaw-rate 上限 `±10 rad/s`，腿长 `0.16–0.33 m`。
- 高度标定：直立时 `base_height ≈ leg_length - 0.012 m`，所以训练高度范围是 `0.148–0.318 m`。
- 一次长训练内的指令课程：前 1500 个 PPO iteration 将 `vx` 从 `±0.5→±2.3 m/s`、运动 yaw 从 `±1→±4 rad/s`、原地 yaw 从 `±2→±10 rad/s`、腿长从 `0.20–0.26→0.16–0.33 m` 连续扩展；不拆成多套 DR 任务，也不做 checkpoint 阶段迁移。
- 高度命令每 5 秒瞬时跳变；15% 样本直接取当前课程范围的最低或最高端点，不使用高度变化率限制。
- DR 从第 0 次更新全部启用：摩擦、恢复系数、base mass/inertia/COM、默认关节位置、VMC Kp/Kd、电机强度和 0–10 ms 动作延迟。
- Actor 观测：Fudan 的 25 维 proprioception × 5 帧；不输入真实线速度。
- Encoder：`125 -> [128,64] -> 3`，用真实三维机体线速度乘 Fudan 的 2.0 观测尺度单独监督。
- Actor/Critic：`[128,64,32]` / `[256,128,64]`。
- PPO：Fudan 参数（48 steps/env、5 epochs、4 minibatches、lr 1e-3 adaptive、gamma 0.99、lambda 0.95、KL 0.005、entropy 0.01）。
- privileged observations 删除了平地无信息的 77 维高度扫描，只保留状态与真实 DR 参数。
- 奖励只有 Fudan 平地配置中的 16 项；没有旧仓库额外的停止、滑移、饱和、倾角 barrier 或高度一致性奖励。

详细逐项对齐见 [TRAINING_DESIGN.md](TRAINING_DESIGN.md)。

## 安装与任务检查

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe -m pip install -e source\infantry_2027
D:\condaenvs\isaacsim510\python.exe scripts\list_envs.py --keyword Infantry
```

## 长训练

本机 8 GB RTX 4060 实测 2048 环境约 2114 simulation steps/s，优于 1024 环境的 1378 steps/s，且完成了完整 PPO 更新，因此默认使用 2048：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Flat-v0 --num_envs 2048 --max_iterations 5000 --headless --run_name long_v1_direct_yaw
```

训练日志位于 `isaaclab_ext/logs/rsl_rl/infantry_2027_v0_flat/`。查看 TensorBoard：

```powershell
D:\condaenvs\isaacsim510\python.exe -m tensorboard.main --logdir D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext\logs\rsl_rl\infantry_2027_v0_flat --port 6006
```

浏览器打开 `http://localhost:6006`。

按 `Ctrl+C` 会保存 `model_interrupted_<iteration>.pt`。续训到总计 5000 次更新（不会额外再训练 5000 次）：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Flat-v0 --num_envs 2048 --max_iterations 5000 --headless --resume --load_run <运行目录名> --checkpoint <checkpoint文件名>
```

项目 checkpoint 额外保存 `completed_iterations` 和 Encoder optimizer 状态；续训按已完成更新数计算，不使用文件名猜测进度。

## 可视化

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\play.py --task Infantry-2027-Flat-Play-v0 --num_envs 1 --checkpoint <model_xxx.pt> --real-time
```

Play 的自动随机指令会立即使用最终训练范围，不会从课程初始范围重新爬升。键盘控制命令：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\play.py --task Infantry-2027-Flat-Play-v0 --num_envs 1 --checkpoint <model_xxx.pt> --real-time --keyboard
```

点击一次 Isaac Sim 视口后使用：

- `I / K`：前进 / 后退，默认目标速度为 `±2.3 m/s`。
- `J / L`：长按累加 direct yaw-rate，完全松开立即归零；运动上限 `±4 rad/s`，原地上限 `±10 rad/s`。
- `U / O`：每次按键事件让 base height 瞬时跳变 `±0.02 m`，范围 `0.148–0.318 m`，对应名义腿长 `0.16–0.33 m`。
- `M`：将前进与转向目标立即清零。
- `N`：把 base height 恢复到默认 `0.233 m`（名义腿长 `0.245 m`）。

这组按键避开 Isaac Sim 常用的 `W/A/S/D`、方向键和 `Q/W/E/R/F` 视口快捷键。可用 `--forward-speed`、`--yaw-acceleration`、`--moving-yaw-limit`、`--point-yaw-limit`、`--base-height`、`--height-step` 覆盖默认控制参数。

命令行选项是连字符形式 `--real-time`，不是 `--real_time`。Windows GUI 会同时加载 Isaac Sim、HDF5 与 PyTorch 的原生 DLL；`play.py` 已在 Kit 启动前固定预加载顺序，避免 `0xc0000139` 自动退出。

不加 `--keyboard` 时为最终范围内的随机指令回放；加入后，随机重采样和 episode 时间上限会被禁用，由上述按键持续控制。

## Fudan 全地形预览

独立审阅场景完整覆盖参考 `plane` 中定义的平地、正/负平滑坡、正/负粗糙坡、上下楼梯、离散障碍、踏石、沟壑和深坑，并按十级难度平铺：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\visualize_fudan_terrains.py
```

该窗口不锁定视角。若另一个 Isaac Sim/Isaac Lab 进程仍在运行，请先关闭它再启动预览，避免 Kit 缓存锁冲突。实现及列顺序详见 [isaaclab_ext/README.md](isaaclab_ext/README.md)。

## Terrain-v0 地形训练

`Infantry-2027-Terrain-v0` 只采用 Fudan 参考仓库最终实际参与训练的集合，而不是把预览中的所有极端分支都塞进训练：

- 平地 50%。
- 正/负平滑坡各 10%。
- 正/负粗糙坡各 5%。
- 下/上楼梯各 10%。
- 10 个难度等级严格为 `difficulty = row / 10 = 0.0 ... 0.9`，初始等级不超过 5。

Actor 仍是 25 维 proprioception 的五帧历史和监督三维线速度 Encoder，不读取地形高度；只有 critic 新增机体附近 `11×7=77` 维高度扫描。奖励、课程升级、扰动和 PPO 参数按 Fudan 最终 terrain 配置移植，VMC 仅替代参考仓库的开链关节 PD 执行层。

首先从随机初始化做短测，这是主方案：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Terrain-v0 --num_envs 128 --max_iterations 200 --headless --run_name terrain_v0_scratch_smoke
```

短测通过后正式训练也从零开始，不续训平地模型；环境数需根据 8 GB 显存实测，地形射线会比平地占用更多显存：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Terrain-v0 --num_envs 1024 --max_iterations 5000 --headless --run_name terrain_v0_scratch_long
```

2026-08-24 的最终配置从零短测已通过。200 轮诊断 checkpoint 位于：

```text
isaaclab_ext\logs\rsl_rl\infantry_2027_v0_terrain\2026-08-24_12-06-55\model_200.pt
```

当前正式长训练同样从随机初始化启动，不加载平地或短测权重：

```text
isaaclab_ext\logs\rsl_rl\infantry_2027_v0_terrain\2026-08-24_12-33-28_terrain_v0_scratch_long_5000
```

Terrain runner 每 20 轮保存一次。若训练中断，按“总目标轮数”续训到 5000：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Terrain-v0 --num_envs 1024 --max_iterations 5000 --headless --resume --load_run 2026-08-24_12-33-28_terrain_v0_scratch_long_5000 --checkpoint model_<最近轮数>.pt --kit_args "--portable-root D:/rm/2026_code/rl/infantry_2027_rl/isaaclab_ext/.tmp/kit_terrain_long_5000_resume"
```

只有当短测明确表明随机初始化无法先学会站立与低级地形时，才采用平地模型的 actor+Encoder 权重作为初始化；不直接恢复 optimizer、critic 或训练 iteration，避免把平地 value function 和优化器状态带进新的地形分布。

固定某个地形做可视化（checkpoint 必须来自 Terrain-v0，不能使用 68 维 critic 的 Flat checkpoint）：

```powershell
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\play.py --task Infantry-2027-Terrain-Play-v0 --num_envs 1 --checkpoint <terrain_model_xxx.pt> --terrain-type stairs_up --terrain-level 5 --real-time --keyboard
```

`--terrain-type` 可取 `flat`、`smooth_up`、`smooth_down`、`rough_up`、`rough_down`、`stairs_down`、`stairs_up`；`--terrain-level` 为 0～9。固定预览会关闭课程升级，因此 reset 后仍停留在所选地形。

如果 Kit 报 `user.config.json`/`DerivedDataCache` 被锁，可关闭其他 Isaac Sim 进程，或给当前进程使用隔离目录：

```powershell
--kit_args "--portable-root D:/rm/2026_code/rl/infantry_2027_rl/isaaclab_ext/.tmp/kit_portable"
```

## 当前 v1 平地到地形路线（2026-08-25）

旧 `v0` 任务与 checkpoint 已冻结。新的正式路线使用
`Infantry-2027-Flat-Compatible-v1` 从零训练 2000 updates，再以完整
checkpoint 续训 `Infantry-2027-Terrain-v1`。两阶段 Actor 都是 125 维、
Critic 都是 145 维，完整设计、奖励对齐、中立摆角标定、正/倒车地形
约束、验证结果和命令统一记录在
[V1_FLAT_TO_TERRAIN_PLAN.md](V1_FLAT_TO_TERRAIN_PLAN.md)。

## 服务器一键平地从零训练

服务器工作副本位于 `/root/gpufree-data/rl/infantry_2027_rl`。同步到最新、确认
`git status --short` 为空并安装 editable package 后，在服务器执行：

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
bash -n scripts/automation/start_flat_scratch_server.sh
bash scripts/automation/start_flat_scratch_server.sh
```

脚本固定从随机网络权重启动正式 `Infantry-2027-Flat-Compatible-v1`：4096 环境、
总计 2000 updates。它会拒绝脏工作树和重复训练进程，检查不可变资产与运行时版本，
然后在当前终端用 `exec python -u scripts/rsl_rl/train.py ...` 前台启动。所有训练信息
直接显示在 tmux 窗口中，不使用 `nohup`、输出重定向或后台 PID。该入口只训练平地，
不会自动进入地形；平地结束并分析通过后再显式启动地形阶段。

推荐先创建 tmux 会话再运行：

```bash
tmux new -s flat_v1
```

在 tmux 中执行上面的启动脚本。训练开始后：

```bash
# 脱离但不停止训练：先按 Ctrl-b，再按 d

# 重新进入训练终端
tmux attach -t flat_v1

# 查看所有会话
tmux ls
```

在训练终端按 `Ctrl+C` 会走 `train.py` 的安全中断逻辑并保存
`model_interrupted_<iteration>.pt`；不要用 `kill -9`。

本地代码提交后同时推送 GitHub 和服务器 bare remote：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl
git push origin main
git push server main
```

随后服务器工作副本更新：

```bash
cd /root/gpufree-data/rl/infantry_2027_rl
git pull --ff-only
python -m pip install -e isaaclab_ext/source/infantry_2027
```

训练运行期间不更新服务器工作副本。需要改训练代码时，先让当前训练结束或显式停止，
本地提交并同步后再启动新的正式 run。
