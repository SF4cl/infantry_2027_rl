# infantry_2027 Isaac Lab external project

该仓库由 Isaac Lab 官方 external-project generator 创建，用于训练高保真闭链资产 `infantry_2027_v0`。

完整结构、训练契约、安装、长训练、续训和可视化命令见上一级 [README.md](../README.md)，逐项设计依据见 [TRAINING_DESIGN.md](../TRAINING_DESIGN.md)。

快速启动：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe -m pip install -e source\infantry_2027
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Flat-v0 --num_envs 2048 --max_iterations 5000 --headless --run_name long_v0
```

## Fudan 全地形预览

`scripts\visualize_fudan_terrains.py` 独立复刻 `ref\fudan_rl_wheel_leg\plane` 的全部地形生成分支，不会改变当前平地训练环境。参考代码中的平滑坡和粗糙坡各自包含正、负方向，因此九类生成分支在预览中展开为 11 个可见列；每列包含 `difficulty = 0.0 ... 0.9` 十级难度。

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe scripts\visualize_fudan_terrains.py
```

地图采用参考仓库的参数：单块 `8 m × 8 m`、水平采样 `0.1 m`、垂直采样 `0.005 m`、坡度阈值 `0.75`。启动后 X 轴是难度，Y 轴依次是：平地、正/负平滑坡、正/负粗糙坡、下/上楼梯、离散障碍、踏石、沟壑、深坑。各列使用不同颜色，初始相机只设置一次，之后视角可以自由移动。

脚本默认仅创建完整分辨率的视觉网格，不进行 PhysX 碰撞烹饪；这是地形审阅工具，不是训练入口。请避免同时启动另一份 Isaac Sim/Isaac Lab 进程，否则 Kit 用户配置和缓存锁可能使第二个窗口长时间停在初始化阶段。

## Terrain-v0 正式训练

Terrain-v0 的训练集合、奖励、课程、固定地形可视化和续训命令统一记录在上一级 [README.md](../README.md)。2026-08-24 已完成 200 updates 从零门槛测试，并启动 1024 环境、5000 updates 的正式从零训练：

```text
logs\rsl_rl\infantry_2027_v0_terrain\2026-08-24_12-33-28_terrain_v0_scratch_long_5000
```

正式 run 不加载 Flat checkpoint，也不加载 200 轮诊断 checkpoint；每 20 轮保存一次，便于安全续训。

## Ubuntu 服务器迁移与 v1 续训

当前正式 v1 训练环境与服务器镜像版本契约为：

- Isaac Sim `5.1.0`
- Isaac Lab `v2.3.2`
- Python `3.11`
- 从 `infantry_2027_rl` 根目录整体迁移，不能只复制 `isaaclab_ext`；不可变资产快照位于根目录的 `assets/infantry_2027_v0`

建议服务器至少提供 100GB 可用持久化空间。RTX 4090 24GB、12 核 CPU、50GB RAM 足以先保持平地 4096 环境和地形 1024 环境；迁移后先保持环境数不变，以免同时改变硬件和训练 batch 语义。

假设项目传到 `/workspace/infantry_2027_rl`，先安装外部项目并确认任务注册：

```bash
cd /workspace/infantry_2027_rl/isaaclab_ext
python -m pip install -e source/infantry_2027
python scripts/list_envs.py
```

把完整 `model_850.pt` 及当前 run 的 TensorBoard event 文件复制到服务器。以下命令会从 checkpoint 内记录的 851 个已完成 updates 接着训练，并把 `--max_iterations 2000` 解释为绝对终点，而不是额外再训练 2000 轮：

```bash
mkdir -p logs/automation/flat_to_terrain_v1
nohup python scripts/rsl_rl/train.py \
  --task Infantry-2027-Flat-Compatible-v1 \
  --num_envs 4096 \
  --max_iterations 2000 \
  --headless \
  --run_name flat_compatible_v1_server_resume851 \
  --resume_path /workspace/checkpoints/model_850.pt \
  > logs/automation/flat_to_terrain_v1/flat_server_stdout.log 2>&1 &
echo $! > logs/automation/flat_to_terrain_v1/flat_server.pid
```

训练创建新 run 目录后，启动跨平台自动验收与地形接力。`<flat-run>` 必须替换为本次服务器续训新建的 run 目录，而不是 Windows 上的原 run：

```bash
nohup python scripts/automation/flat_to_terrain_monitor.py \
  --flat-pid "$(cat logs/automation/flat_to_terrain_v1/flat_server.pid)" \
  --flat-run "<flat-run>" \
  --project "$PWD" \
  --python "$(command -v python)" \
  --flat-iterations 2000 \
  --terrain-iterations 5000 \
  --terrain-envs 1024 \
  --status logs/automation/flat_to_terrain_v1/status.json \
  > logs/automation/flat_to_terrain_v1/monitor_server_stdout.log 2>&1 &
```

服务器端查看实时进度：

```bash
tail -f logs/automation/flat_to_terrain_v1/flat_server_stdout.log
watch -n 10 cat logs/automation/flat_to_terrain_v1/status.json
nvidia-smi
```
