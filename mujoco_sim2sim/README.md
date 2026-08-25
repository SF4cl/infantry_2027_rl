# infantry_2027_v0 MuJoCo sim2sim

最终 checkpoint 的 ONNX 导出、125 维输入、6 维输出、坐标/关节方向、500 Hz
VMC 和单片机部署契约见 [POLICY_ONNX_SIM2REAL.md](POLICY_ONNX_SIM2REAL.md)。

这里部署的是修改转向/高度命令设计之前的基线模型：

- checkpoint：`model_1600.pt`
- 训练记录：`2026-08-23_17-26-24_long_v0`
- checkpoint schema：`infantry-2027-v0-fudan-estimator`
- policy：5 帧 × 25 维本体观测，三维监督速度 Encoder，6 维 VMC 动作
- 控制频率：MuJoCo 500 Hz，policy 100 Hz
- 资产：直接读取不可变快照 `../assets/infantry_2027_v0/mujoco/infantry_2027_v0.xml`

运行时不会修改资产快照。脚本在内存中为 MJCF 增加地面，并关闭机器人内部刚体碰撞，以对应 Isaac Lab 中的 `enabled_self_collisions=False`。

## 环境与导出

以下命令均在 `D:\rm\2026_code\rl\infantry_2027_rl` 下执行。

重新导出 checkpoint（一般不需要重复执行）：

```powershell
D:\condaenvs\isaacsim510\python.exe .\mujoco_sim2sim\scripts\export_policy.py `
  --checkpoint .\isaaclab_ext\logs\rsl_rl\infantry_2027_v0_flat\2026-08-23_17-26-24_long_v0\model_1600.pt `
  --output .\mujoco_sim2sim\exported\model_1600.npz
```

导出文件带有 checkpoint SHA-256、迭代数以及 PyTorch/NumPy 固定输入输出自检。

## 可视化

固定的 0.8 m/s 前进指令：

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --viewer --scenario forward --duration 120
```

键盘交互：

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --viewer --keyboard --duration 600
```

键位使用数字小键盘，避免和 MuJoCo 自带快捷键冲突。控制状态机已经与最新 Isaac Lab `play.py` 对齐：

- `Num 8 / Num 2`：按住时直接给定前进 / 后退目标，松开立即归零
- `Num 4 / Num 6`：按住时以 10 rad/s² 累加左 / 右转向，松开两键立即归零
- `Num 7 / Num 1`：每次按下让目标高度瞬间增加 / 减少 0.02 m；长按不会连续爬升
- `Num 5`：按住时速度和转向归零
- `Num 0`：把目标高度重置为初始值 0.233 m

MuJoCo 的 passive viewer 回调不提供按键松开事件，因此脚本使用 Windows `GetAsyncKeyState` 轮询真实按键状态。窗口打开后应让 MuJoCo 窗口保持焦点并开启 `Num Lock`。

这份旧 `model_1600` 的训练转向范围是 ±3 rad/s，默认参数因此为 `--moving-yaw-limit 3 --point-yaw-limit 3`。它只同步最新按键逻辑，不会向旧模型发送后来训练才加入的 10 rad/s 原地转指令。

自定义命令的顺序是 `vx yaw_rate base_height`。例如：

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --viewer --command 0.8 0.8 0.215 --duration 120
```

这个旧模型采用旧训练阶段的有效 yaw-rate 输入，没有专门训练后来增加的高速原地转模式。因此此处只把运动转向作为有效验收项，不把零前进速度下的大 yaw-rate 当作该 checkpoint 的能力。

## 已验证结果

无界面 smoke test 使用 nominal 执行器参数、零动作延迟和原始 MuJoCo 质量/惯量：

| 工况 | 是否存活 | 实际均值/误差 | 倾角 P90 | 最大闭链残差 |
|---|---:|---:|---:|---:|
| 站立，height=0.215 m | 是 | vx=-0.133 m/s | 0.41° | 0.298 mm |
| 前进，vx=0.8 m/s | 是 | vx=0.531 m/s，MAE=0.269 m/s | 1.11° | 0.308 mm |
| 前进左转，0.8 m/s、0.8 rad/s | 是 | yaw=0.735 rad/s，MAE=0.065 rad/s | 0.91° | 0.355 mm |
| 最低高度，height=0.148 m | 是 | height=0.158 m | 1.13° | 0.326 mm |
| 最高高度，height=0.318 m | 是 | height=0.302 m | 1.00° | 0.407 mm |

结论：policy、25 维观测、Encoder、VMC、闭链和左右符号映射已经在 MuJoCo 中闭环跑通；没有翻倒、数值发散或闭链失稳。当前仍存在清晰的 sim2sim 差异：静止后向漂移、前进速度欠跟踪、高腿长静止漂移增大。因此这是“基础 sim2sim 通过”，不是最终部署验收通过。

## 无界面复现

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py --scenario stand --duration 5 --settle 1
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py --scenario forward --duration 6 --settle 1
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py --scenario forward_left --duration 6 --settle 1
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py --command 0 0 0.148 --duration 5 --settle 1
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py --command 0 0 0.318 --duration 5 --settle 1
```
