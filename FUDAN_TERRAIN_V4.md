# Fudan-aligned direct-joint terrain v4

`Infantry-2027-Joint-Fudan-Terrain-v4` 以参考仓库最终实际训练快照
`plane/logs/wheel_legged/6010.2加入随机地形terrain_proportions0.50.20.10.10.10.0`
为唯一基准。它从随机网络权重直接训练平地与地形，不加载 v3 或任何平地
checkpoint。

## 与参考实现相同的训练契约

- 4096 个并行环境，100 Hz policy，20 s episode，48 steps/env。
- 20 个地形列、10 个难度行，`difficulty = row / 10`；初始 level 在 0～5
  均匀随机。
- 地形占比：平地 50%，正/负平滑坡各 10%，正/负粗糙坡各 5%，下/上楼梯
  各 10%。
- 行进距离超过 4 m 升一级；未升级且线速度跟踪得分低于 0.4 时降一级；超过
  level 9 后随机回到 0～9。
- 每个环境独立维护命令范围。初始 `vx=[-2,2] m/s`、direct
  `yaw-rate=[-2,2] rad/s`，每 10 s 均匀重采样一次；没有强制 heading、特殊
  正反向分布、站立/原地转模式或按全局轮数扩展命令。
- level 0 失败时，`vx` 两侧各收缩 0.25、yaw 两侧各收缩 0.5，最小绝对范围
  均为 1.0。通过 level 9 且线速度和 yaw 跟踪均大于 0.7 时，基础地形的
  `vx/yaw` 两侧各增长 0.5，楼梯向上的两侧各增长 0.05/0.1；最终分别限制为
  `±2.5 m/s` 和 `±4 rad/s`。
- 六维动作直接控制四个腿关节的位置 PD 与两个轮关节的速度 PD：腿
  `scale=0.5, Kp=60, Kd=1`，轮 `scale=10, Kd=0.2`，动作延迟 0～10 ms。
- Actor 为 5 帧 × 25 维本体历史；Encoder 监督三维机体系线速度；critic 为
  145 维 privileged observation，其中包含 77 维地形高度扫描。
- 奖励、逐项裁剪、完整 DR、持续倒地终止与 PPO 参数均使用上述最终快照；
  policy 初始噪声为 1.0，默认 50000 updates，每 100 updates 保存一次。

## 必要的资产/仿真适配

参考模型与 `infantry_2027_v0` 的几何和关节定义不同，因此仅保留以下不可避免
的差异：

- 使用经过验证的新闭链资产、质量、惯量、关节限位和执行器力矩限位。
- 参考 base-height 下限 0.14 m 超出新资产已验证的 0.16 m 最短腿长，因此
  只把下限夹到 0.148 m；上限仍按参考为 0.300 m。
- 物理步长采用已验证闭链资产的 `dt=0.002, decimation=5`，与参考
  `dt=0.005, decimation=2` 具有相同的 100 Hz policy 频率。
- 保留 non-finite 安全终止；它只拦截数值故障，不改变正常训练目标。

## 从零训练

服务器上建议在 tmux 中直接运行：

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
python -u scripts/rsl_rl/train.py \
  --task Infantry-2027-Joint-Fudan-Terrain-v4 \
  --num_envs 4096 \
  --max_iterations 50000 \
  --headless \
  --run_name joint_fudan_terrain_v4_4096x50000_server_scratch
```

命令中不得加入 `--resume`。若只计划先观察 10000 updates，可把
`--max_iterations` 改为 10000；训练目标和其他配置不变。

## 可视化

```bash
cd /root/gpufree-data/rl/infantry_2027_rl/isaaclab_ext
python scripts/rsl_rl/play.py \
  --task Infantry-2027-Joint-Fudan-Terrain-Play-v4 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model_xxx.pt \
  --terrain-type stairs_up \
  --terrain-level 5 \
  --real-time \
  --keyboard
```

固定地形时可用 `flat`、`smooth_up`、`smooth_down`、`rough_up`、
`rough_down`、`stairs_down`、`stairs_up`。默认视角跟随机器人平移；加
`--free-camera` 可恢复自由相机。
