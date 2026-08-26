# infantry_2027 Isaac Lab external project

该仓库由 Isaac Lab 官方 external-project generator 创建，用于训练高保真闭链资产 `infantry_2027_v0`。

完整结构、训练契约、安装、长训练、续训和可视化命令见上一级 [README.md](../README.md)，逐项设计依据见 [TRAINING_DESIGN.md](../TRAINING_DESIGN.md)。

快速启动：

```powershell
cd D:\rm\2026_code\rl\infantry_2027_rl\isaaclab_ext
D:\condaenvs\isaacsim510\python.exe -m pip install -e source\infantry_2027
D:\condaenvs\isaacsim510\python.exe scripts\rsl_rl\train.py --task Infantry-2027-Flat-Stable-v2 --num_envs 2048 --max_iterations 5000 --headless --run_name flat_stable_v2_local_scratch
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

Terrain-v0 的训练集合、奖励、课程、固定地形可视化和续训命令统一记录在上一级 [README.md](../README.md)。
训练日志、checkpoint 和训练后指标分析作为外部产物管理，不提交到本仓库。

## Ubuntu 服务器 v2 从零训练

当前正式 v2 训练环境与服务器镜像版本契约为：

- Isaac Sim `5.1.0`
- Isaac Lab `v2.3.2`
- Python `3.11`
- 从 `infantry_2027_rl` 根目录整体迁移，不能只复制 `isaaclab_ext`；不可变资产快照位于根目录的 `assets/infantry_2027_v0`

建议服务器至少提供 100GB 可用持久化空间。RTX 4090 24GB、12 核 CPU、50GB RAM 足以先保持平地 4096 环境和地形 1024 环境；迁移后先保持环境数不变，以免同时改变硬件和训练 batch 语义。

服务器项目位于 `/root/gpufree-data/rl/infantry_2027_rl`。同步 Git 后安装外部项目：

```bash
cd /root/gpufree-data/rl/infantry_2027_rl
git pull --ff-only
python -m pip install -e isaaclab_ext/source/infantry_2027
```

正式平地从随机网络权重开始，不加载旧 checkpoint。创建 tmux 会话：

```bash
tmux new -s flat_v2
```

在 tmux 中前台执行：

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
bash -n scripts/automation/start_flat_scratch_server.sh
bash scripts/automation/start_flat_scratch_server.sh
```

脚本检查资产、Git 工作树、重复进程和运行时版本，然后使用 `exec` 在当前终端直接运行
`train.py`。所有训练信息原样显示在 tmux 中；没有 `nohup`、后台 PID、日志重定向或
自动地形接力。脱离与恢复 tmux：

```bash
# 先按 Ctrl-b，再按 d 脱离
tmux attach -t flat_v2
tmux ls
```

训练完成后先分析平地 `model_5000.pt`，通过后再建立并单独启动 Terrain-v2。训练终端内按
`Ctrl+C` 会安全保存 `model_interrupted_<iteration>.pt`。
