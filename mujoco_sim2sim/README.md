# infantry_2027_v0 MuJoCo sim2sim

该目录只保留可复现的 policy 导出、闭环 MuJoCo 运行时、VMC 和可视化入口。
参数扫描、绘图、一次性诊断代码及生成的表格/图片保存在仓库外的本机分析工作区。

完整的 ONNX 输入输出、坐标系、关节方向、500 Hz VMC 和实机接口契约见
[POLICY_ONNX_SIM2REAL.md](POLICY_ONNX_SIM2REAL.md)。

## 运行契约

- 资产：`../assets/infantry_2027_v0/mujoco/infantry_2027_v0.xml`。
- MuJoCo 物理频率：500 Hz；policy 频率：100 Hz。
- policy：5 帧 × 25 维本体观测，三维监督速度 Encoder，6 维 VMC 动作。
- 运行时不修改不可变资产快照；地面和与 Isaac Lab 一致的自碰撞设置在内存中构建。

checkpoint、导出的 NPZ/ONNX、日志和结果文件均为外部产物，不进入 Git。

## 导出 policy

在项目根目录执行：

```powershell
D:\condaenvs\isaacsim510\python.exe .\mujoco_sim2sim\scripts\export_policy.py `
  --checkpoint <model_xxx.pt> `
  --output .\mujoco_sim2sim\exported\<policy_name>.npz
```

导出文件包含 checkpoint SHA-256、iteration、schema 和固定输入输出自检向量。

## 可视化

固定命令：

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --policy .\mujoco_sim2sim\exported\<policy_name>.npz `
  --viewer --command 0.8 0.0 0.215 --duration 120
```

键盘交互：

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --policy .\mujoco_sim2sim\exported\<policy_name>.npz `
  --viewer --keyboard --duration 600
```

数字小键盘避开 MuJoCo 自带快捷键：

- `Num 8 / Num 2`：按住时给定前进 / 后退，松开归零。
- `Num 4 / Num 6`：按住累加左 / 右 yaw-rate，松开归零。
- `Num 7 / Num 1`：目标高度瞬时增加 / 减少。
- `Num 5`：速度和转向归零。
- `Num 0`：恢复默认目标高度。

MuJoCo passive viewer 不提供松键回调，Windows 下通过 `GetAsyncKeyState` 轮询按键状态。
运行时请保持 MuJoCo 窗口焦点并开启 `Num Lock`。

## 无界面复现

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --policy .\mujoco_sim2sim\exported\<policy_name>.npz `
  --scenario stand --duration 5 --settle 1

D:\rm\2026_code\rl\condaenvs\mujoco\python.exe .\mujoco_sim2sim\scripts\run_policy.py `
  --policy .\mujoco_sim2sim\exported\<policy_name>.npz `
  --command 0.8 0.8 0.215 --duration 6 --settle 1
```

如果需要保存 JSON 摘要，使用 `--output <仓库外的路径>`，不要将结果提交到 Git。
