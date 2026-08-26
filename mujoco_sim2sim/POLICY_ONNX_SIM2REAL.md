# infantry_2027 policy ONNX 与实机接口契约

本文档对应 policy contract `infantry-2027-v0-flat-25d-v1`，用于将训练
checkpoint 导出为 ONNX，并在 PC、嵌入式计算机或单片机上复现与 Isaac Lab
一致的 Encoder、Actor、观测历史和 VMC 控制。

本文档中的数组索引全部从 0 开始。角度单位为 rad，角速度为 rad/s，长度为 m，
线速度为 m/s，力矩为 N·m。除非特别说明，网络张量均为 IEEE-754 float32。

## 1. 文件和导出命令

导出脚本：

```text
mujoco_sim2sim/scripts/export_policy_onnx.py
```

以最终 `model_5000.pt` 为例，在仓库根目录运行：

```powershell
D:\condaenvs\isaacsim510\python.exe .\mujoco_sim2sim\scripts\export_policy_onnx.py `
  --checkpoint .\isaaclab_ext\logs\rsl_rl\infantry_2027_v0_flat\2026-08-24_00-53-14_long_v1_direct_yaw\model_5000.pt `
  --action-contract infantry-2027-v0-vmc-action-v1 `
  --output .\mujoco_sim2sim\exported\model_5000.onnx
```

导出的 ONNX 与 PyTorch Encoder/Actor 一样保留任意 batch，batch 维不会固定为 1。
脚本会拒绝错误的 checkpoint schema 或网络维度，并生成：

```text
model_5000.onnx          Encoder + deterministic Actor
model_5000.onnx.json     checkpoint/ONNX SHA-256、I/O 和验证结果
model_5000.golden.json   固定跨平台测试向量
```

ONNX 内嵌 checkpoint SHA-256、迭代数、policy/action contract、平衡摆角表、坐标系和 I/O 描述。
部署时应校验 manifest 中的哈希，禁止仅凭文件名判断 policy 版本。

## 2. ONNX 图的精确接口

模型接口为：

```text
input:
  history                         float32[batch, 125]

outputs:
  actions                         float32[batch, 6]
  estimated_base_lin_vel_scaled   float32[batch, 3]
```

`batch` 是动态维度。实机单机器人运行时通常传入 `[1,125]`，PC 批量检查时可直接
传入 `[N,125]`；二者使用同一个 ONNX 文件。建议按 tensor 名称取输出，不要只
依赖返回顺序。

ONNX 只包含：

```text
history -> Encoder -> latent(3)
history 最后 25 维 + latent -> Actor -> raw actions(6)
```

ONNX 不包含以下运行时逻辑：

- 观测构造、缩放和 5 帧历史维护；
- raw action 的 `[-100, 100]` 裁剪；
- 腿角、腿长和轮速解码；
- 五连杆正运动学、Jacobian 和 VMC；
- 500 Hz PD、前馈、力矩限制及电机通信；
- 指令限幅、急停和安全状态机。

因此 `actions` 不能直接当成 6 路电机力矩。

Encoder 生成 3 维速度诊断口，其训练监督目标为：

```text
estimated_base_lin_vel_scaled = 2.0 * [vx_body, vy_body, vz_body]
```

Actor 在 ONNX 图内部使用同一个 latent。`estimated_base_lin_vel_scaled` 额外导出
用于联调和诊断，不参与单片机外部反馈；真正的控制输出仍然是 6 维 `actions`。

网络共有 38,825 个 float32 参数，约 151.7 KiB 原始权重，单次推理约 38,400
个 MAC。固定图只使用 `Gemm`、`Elu`、`Slice` 和 `Concat` 等基础算子。若 MCU
工具链不直接运行 ONNX，应使用芯片厂商的 ONNX 转换器生成本地网络代码，并用
golden vector 做转换后的逐值验收。

## 3. 频率和数据时序

物理/VMC 内环为 500 Hz，周期 2 ms；policy 为 100 Hz，周期 10 ms。每个
history 含 5 个 25 维采样，按最老到最新排列：

| history 范围 | 时刻 | 含义 |
|---:|---:|---|
| `[0, 24]` | `t-40 ms` | 最老帧 |
| `[25, 49]` | `t-30 ms` |  |
| `[50, 74]` | `t-20 ms` |  |
| `[75, 99]` | `t-10 ms` |  |
| `[100, 124]` | `t` | 当前帧，Actor 实际使用这一帧 |

5 个样本覆盖 40 ms 的首尾时间跨度。初始化或控制器复位时，必须先构造一帧
有效观测，再把这一帧复制到全部 5 个位置；不能初始化成四帧零加一帧当前值。

100 Hz 周期的推荐次序：

```text
1. 读取同一时间基准下的 IMU、4 个腿编码器和 2 个轮编码器；
2. 计算机体系角速度和 projected gravity；
3. 用“上一个 100 Hz 周期输出的 raw action”构造当前 25 维 frame；
4. history 左移 25 个 float，并把当前 frame 写入 [100,124]；
5. ONNX 推理得到 new_action；
6. 检查 finite，裁剪到 [-100,100]；
7. 保存为 previous_raw_action，交给 500 Hz 内环保持执行。
```

`previous_raw_action` 是 Actor 的原始输出，不是解码后的腿长、腿角、轮速或力矩。

## 4. 单帧 25 维观测

### 4.1 完整索引表

| 帧内索引 | history 当前帧索引 | 数据 | 单位/缩放 | 正方向或说明 |
|---:|---:|---|---|---|
| 0 | 100 | `base_ang_vel_x` | `rad/s * 0.25` | 绕 `+X` |
| 1 | 101 | `base_ang_vel_y` | `rad/s * 0.25` | 绕 `+Y` |
| 2 | 102 | `base_ang_vel_z` | `rad/s * 0.25` | 绕 `+Z`，正值左转 |
| 3 | 103 | `projected_gravity_x` | 无量纲 | 世界重力在机体系的 X 分量 |
| 4 | 104 | `projected_gravity_y` | 无量纲 | 世界重力在机体系的 Y 分量 |
| 5 | 105 | `projected_gravity_z` | 无量纲 | 水平静止时约为 -1 |
| 6 | 106 | `command_vx` | `m/s * 2.0` | 正值前进 |
| 7 | 107 | `command_yaw_rate` | `rad/s * 0.25` | 正值左转 |
| 8 | 108 | `command_base_height` | `m * 5.0` | base_link 原点的世界 Z 目标 |
| 9 | 109 | `q_lf - q_lf_default` | rad | 左前主动腿物理关节坐标 |
| 10 | 110 | `q_lb - q_lb_default` | rad | 左后主动腿物理关节坐标 |
| 11 | 111 | `q_rf - q_rf_default` | rad | 右前主动腿物理关节坐标 |
| 12 | 112 | `q_rb - q_rb_default` | rad | 右后主动腿物理关节坐标 |
| 13 | 113 | `qd_lf` | `rad/s * 0.05` | 左前主动腿物理速度 |
| 14 | 114 | `qd_lb` | `rad/s * 0.05` | 左后主动腿物理速度 |
| 15 | 115 | `qd_lw` | `rad/s * 0.05` | 左轮；正值对应前进旋转 |
| 16 | 116 | `qd_rf` | `rad/s * 0.05` | 右前主动腿物理速度 |
| 17 | 117 | `qd_rb` | `rad/s * 0.05` | 右后主动腿物理速度 |
| 18 | 118 | `qd_rw` | `rad/s * 0.05` | 右轮；负值对应前进旋转 |
| 19 | 119 | `previous_raw_action[0]` | 无 | 左腿角 action |
| 20 | 120 | `previous_raw_action[1]` | 无 | 左腿长残差 action |
| 21 | 121 | `previous_raw_action[2]` | 无 | 左轮 action |
| 22 | 122 | `previous_raw_action[3]` | 无 | 右腿角 action |
| 23 | 123 | `previous_raw_action[4]` | 无 | 右腿长残差 action |
| 24 | 124 | `previous_raw_action[5]` | 无 | 右轮 action |

前四帧使用相同的帧内顺序，只需分别加上 history 起始偏移 0、25、50、75。

### 4.2 Policy 没有输入的量

Actor/Encoder 不输入：

- 真实或外部定位得到的机体线速度；
- accelerometer 原始线加速度；
- 世界位置、航向角或里程计位置；
- 实际 base 高度；
- 腿长、腿角或电机力矩；
- 电池电压和电机温度。

索引 8 是目标 base height，不是测量高度。Encoder 从 5 帧本体状态中估计机体
线速度。

### 4.3 训练噪声含义

训练时在已经缩放的帧上加入独立均匀噪声：

| 数据 | 缩放后噪声幅值 | 等效原始幅值 |
|---|---:|---:|
| 三轴角速度 | `+/-0.05` | `+/-0.2 rad/s` |
| projected gravity | `+/-0.05` | 同左 |
| 四个腿关节位置 | `+/-0.02` | `+/-0.02 rad` |
| 六个关节速度 | `+/-0.075` | `+/-1.5 rad/s` |
| 命令、previous action | 0 | 0 |

实机推理不要再主动叠加随机噪声。这里的训练噪声是为了提高对真实传感器噪声的
鲁棒性，不是部署算法的一部分。

## 5. 机体坐标系和旋转正方向

机器人采用右手坐标系：

```text
+X：车头/前方
+Y：机器人左方
+Z：上方
```

依照右手定则：

- `+roll/+omega_x`：绕前方轴旋转，机器人左侧抬高；
- `+pitch/+omega_y`：绕左方轴旋转，车头向下；
- `+yaw/+omega_z`：从上方看逆时针，机器人向左转。

若 `R_WB` 表示把机体系向量旋转到世界系，则：

```text
projected_gravity = transpose(R_WB) * [0, 0, -1]
```

水平静止时必须得到：

```text
[0, 0, -1]
```

不要把 IMU 的原始安装坐标直接送入 policy。必须先应用固定安装旋转，把 gyro 和
姿态变换到上述 base_link 坐标。不同库的四元数 `wxyz/xyzw` 排列不同，建议先转
成旋转矩阵，再使用上式，避免在接口中猜测四元数顺序。

## 6. 关节名称、物理轴和左右符号

`f` 表示五连杆前侧主动臂，`b` 表示后侧主动臂，`l/r` 表示机器人左/右：

| 关节 | MJCF/URDF 物理轴 | 观测中使用 | 统一 VMC 坐标 |
|---|---|---|---|
| `lf_joint` | `+Y` | 原始物理 `q_lf` | `q_lf` |
| `lb_joint` | `+Y` | 原始物理 `q_lb` | `q_lb` |
| `rf_joint` | `-Y` | 原始物理 `q_rf` | `-q_rf` |
| `rb_joint` | `-Y` | 原始物理 `q_rb` | `-q_rb` |
| `lw_joint` | `-Y` | 原始物理 `q/qd_lw` | 正速度为前进 |
| `rw_joint` | `+Y` | 原始物理 `q/qd_rw` | 负速度为前进 |

特别注意：右腿关节在“观测数组”中不能取反。只有五连杆运动学和 VMC 内部使用：

```text
left canonical q/qd  =  left physical q/qd
right canonical q/qd = -right physical q/qd
```

实机编码器转换建议显式写成：

```text
q_model  = encoder_position / gear_ratio * encoder_sign - zero_offset
qd_model = encoder_velocity / gear_ratio * encoder_sign
```

这里得到的 `q_model` 必须已经符合上表的物理关节轴；不能再根据左右侧额外取反。
观测使用 `q_model - q_default`。当前资产部署 nominal default 为 0 rad，训练时仅用
`+/-0.03 rad` 随机化增强零位鲁棒性。

## 7. 六维 raw action 和物理含义

ONNX `actions[0, :]` 的顺序固定为：

```text
[left_angle,
 left_length_residual,
 left_wheel,
 right_angle,
 right_length_residual,
 right_wheel]
```

首先执行：

```text
raw[i] = clip(actions[i], -100, 100)
```

然后解码：

| 索引 | 含义 | 解码公式 | 正方向 |
|---:|---|---|---|
| 0 | 左腿腿角目标 | `theta_L* = 0.5 * raw[0] + equilibrium(L_L*)` | 正值使轮心相对髋关节向后 |
| 1 | 左腿长度残差 | 见下式 | 正值加长 |
| 2 | 左轮归一化轮速 | `omega_L_norm = clip(20*raw[2],-55,55)` | 正值前进 |
| 3 | 右腿腿角目标 | `theta_R* = 0.5 * raw[3] + equilibrium(L_R*)` | 正值使轮心相对髋关节向后 |
| 4 | 右腿长度残差 | 见下式 | 正值加长 |
| 5 | 右轮归一化轮速 | `omega_R_norm = clip(20*raw[5],-55,55)` | 正值前进 |

左右目标腿长：

```text
L_L* = clip(command_base_height + 0.012 + 0.03*tanh(raw[1]), 0.16, 0.33)
L_R* = clip(command_base_height + 0.012 + 0.03*tanh(raw[4]), 0.16, 0.33)
```

所以腿长 action 的物理范围是 nominal length 上下 0.03 m，使用 `tanh` 饱和。
`command_base_height + 0.012` 是名义腿长；base-height 命令本身不是最终腿长。

Stable-v2 对解码后的左右目标腿长分别做线性插值，并把平衡摆角偏置加到对应目标：

```text
length nodes: [0.16, 0.22, 0.28, 0.33] m
angle nodes:  [0.0,  0.0, -0.005, -0.005] rad
```

旧 v0 checkpoint 没有这项偏置。部署产物必须携带明确的 action contract，不能仅凭
网络形状判断应采用哪一种解码。

腿角以竖直向下为 0：

```text
theta = 0：轮心在髋关节正下方
theta > 0：轮心向机器人后方，腿向后倾
theta < 0：轮心向机器人前方，腿向前倾
```

资产 `q=0` 时约为：

```text
leg_length = 0.2276652114 m
leg_angle  = -0.0408183666 rad
```

轮速从统一前进坐标映射到物理关节：

```text
physical_lw_velocity_target = +omega_L_norm
physical_rw_velocity_target = -omega_R_norm
```

因此不能把同号轮速直接下发给左右两个物理轮关节。

## 8. 500 Hz 五连杆 VMC

### 8.1 标定参数

```text
l1          = 0.2150 m
l2          = 0.2537 m
phi1_offset = 2.749420977758278 rad
phi4_offset = 0.31053494255178626 rad
```

每侧 canonical 输入：

```text
phi1 = canonical_front_q + phi1_offset
phi4 = canonical_back_q  + phi4_offset
```

必须复用项目 `vmc/five_bar.py` 或 `mujoco_sim2sim/scripts/vmc.py` 中相同的装配
分支、Jacobian 和奇异判断，不能只按杆长重新写另一套五连杆解算分支。

### 8.2 任务空间控制

训练 nominal 增益：

```text
Kp_length = 900 N/m
Kd_length = 20 N/(m/s)
Kp_angle  = 50 N*m/rad
Kd_angle  = 3 N*m/(rad/s)
support_force = 118.88 N，每条腿
```

每侧计算：

```text
F = Kp_length * (L_target - L)
    - Kd_length * L_rate
    + support_force

M = Kp_angle * wrap_to_pi(theta_target - theta)
    - Kd_angle * theta_rate

tau_canonical = transpose(J) * [F, M]
```

再映射到物理关节：

```text
left physical [tau_lf, tau_lb]  = +tau_canonical_left
right physical [tau_rf, tau_rb] = -tau_canonical_right
```

腿电机最终力矩限制为 `+/-45 N*m`。若五连杆进入 singular 状态，原实现将该侧
两路腿力矩置零；实机安全层还应立即退出 policy 控制并进入受控停机，而不是持续
无力矩等待跌倒。

### 8.3 轮电机速度 PD

训练控制为：

```text
tau_lw = clip(1.0 * (physical_lw_velocity_target - qd_lw), -5, 5)
tau_rw = clip(1.0 * (physical_rw_velocity_target - qd_rw), -5, 5)
```

轮力矩限制 `+/-5 N*m`，统一轮速目标限制 `+/-55 rad/s`。仿真电机速度上限为
60 rad/s。

## 9. 指令范围

最终训练范围：

```text
forward velocity:       [-2.3, 2.3] m/s
moving yaw rate:        [-4.0, 4.0] rad/s
point-turn yaw rate:   [-10.0,10.0] rad/s
base_link height:       [0.148,0.318] m
```

当 `abs(vx) < 0.15 m/s` 时训练命令分布允许原地转向，否则属于运动转向。
base height 是世界系中 base_link 原点的 Z 目标，不是底盘离地间隙、IMU 高度或
腿长。

MuJoCo `runtime.py` 根据 `abs(vx) < 0.15 m/s` 区分原地转向与运动转向，并分别限制为
`+/-10 rad/s` 和 `+/-4 rad/s`。高转速命令仍应逐级完成 sim2sim/HIL 验收后再用于实机。

## 10. MCU 侧参考伪代码

```c
float history[125];
float previous_action[6] = {0};

void policy_reset(const Sensors *s, const Command *cmd) {
    float frame[25];
    build_frame(frame, s, cmd, previous_action);
    for (int k = 0; k < 5; ++k) {
        memcpy(&history[25 * k], frame, 25 * sizeof(float));
    }
}

void policy_tick_100hz(const Sensors *s, const Command *cmd) {
    float frame[25];
    float actions[6];
    float velocity_scaled[3];

    build_frame(frame, s, cmd, previous_action);
    memmove(&history[0], &history[25], 100 * sizeof(float));
    memcpy(&history[100], frame, 25 * sizeof(float));

    onnx_infer(history, actions, velocity_scaled);
    if (!all_finite(actions, 6)) emergency_stop();

    for (int i = 0; i < 6; ++i) {
        previous_action[i] = clamp(actions[i], -100.0f, 100.0f);
    }
    decode_policy_targets(previous_action, cmd->base_height);
}

void vmc_tick_500hz(const Sensors *s) {
    update_five_bar_state_with_right_side_sign_mapping(s);
    compute_leg_vmc_and_wheel_pd();
    apply_torque_speed_joint_and_thermal_limits();
    send_motor_commands();
}
```

单片机必须使用确定性的 100 Hz/500 Hz 时钟。需要记录最坏周期和端到端延迟，
不能只记录平均推理时间。

## 11. Golden vector 验收

`model_5000.golden.json` 含固定 `history[125]`、6 维 action 和 3 维速度诊断值。每一种部署后端都应
执行：

1. 加载完全相同的 125 个 float32；
2. 按输出名称读取 action 和速度诊断值；
3. 与 golden 文件逐项比较；
4. 满足文件中的 `atol` 和 `rtol`；
5. 再用实机采集 history 与 PC ONNX Runtime 做在线双算对比。

量化为 FP16、INT8 或芯片专用低精度网络后不能沿用 FP32 容差。必须建立真实观测
数据集，重新统计 action 误差、闭环稳定性和极端输入误差；未做闭环回归前不要使用
量化网络控制真机。

PC 端可用 ONNX Runtime 做最小接口检查：

```python
import json
import numpy as np
import onnxruntime as ort

golden = json.load(open("model_5000.golden.json", encoding="utf-8"))
history = np.asarray(golden["history"], dtype=np.float32).reshape(1, 125)

session = ort.InferenceSession(
    "model_5000.onnx",
    providers=["CPUExecutionProvider"],
)
actions, velocity_scaled = session.run(
    ["actions", "estimated_base_lin_vel_scaled"],
    {"history": history},
)

np.testing.assert_allclose(
    actions[0], np.asarray(golden["actions"], dtype=np.float32),
    atol=golden["atol"], rtol=golden["rtol"],
)
np.testing.assert_allclose(
    velocity_scaled[0],
    np.asarray(golden["estimated_base_lin_vel_scaled"], dtype=np.float32),
    atol=golden["atol"], rtol=golden["rtol"],
)
```

## 12. 上车前的方向检查

第一次通电必须架空或固定机器人，逐项确认：

| 测试 | 预期 |
|---|---|
| 水平静止 | `projected_gravity ~= [0,0,-1]` |
| 左侧抬高 | 正 roll / gyro X |
| 车头向下 | 正 pitch / gyro Y |
| 人工向左转 | 正 gyro Z |
| `lf/lb` 沿物理 `+Y` 转 | 对应 q 增加 |
| `rf/rb` 沿物理 `-Y` 转 | 对应 q 增加 |
| `lw` 物理速度为正 | 左轮产生前进方向 |
| `rw` 物理速度为负 | 右轮产生前进方向 |
| 左右 canonical 腿角同为正 | 两个轮心都相对髋关节向后 |

任何一项不符合时应修复传感器/执行器映射，不能通过调换 policy action 下标或在
观测中临时增加取反来“试到能走”。

## 13. Sim2real 部署边界和安全要求

当前训练已随机化摩擦、恢复系数、base 质量/惯量/质心、关节默认位置、长度 VMC
Kp/Kd、电机强度和 0～10 ms 动作延迟，并加入本体观测噪声。但尚未完整建模：

- 编码器固定偏置、量化、丢包和速度滤波延迟；
- IMU 安装误差、gyro bias、姿态滤波延迟和时间同步；
- 电机死区、库仑摩擦、回差、减速器效率；
- 力矩-转速-母线电压曲线和左右电机差异；
- 轮胎形变、有效半径、侧向摩擦和地面不平；
- 温升、持续电流和电池电压下降。

进入实机高速测试前，应先辨识这些参数，用实测分布补充仿真并重新训练。当前 policy
是平地任务，全地形仅有预览场景，不能把平地 sim2sim 通过等同于全地形实机能力。

实机控制器至少要独立于 policy 实现：

- 物理急停和通信看门狗；
- IMU/编码器时间戳过期与非有限值检查；
- 腿 `+/-45 N*m`、轮 `+/-5 N*m` 及真实电流/温度限制；
- 主动关节速度和被动膝机械限位监控；
- 倾角、机体高度、闭链/VMC 奇异状态保护；
- 指令斜坡和分级解锁；
- policy 超时后进入零轮速、受控腿长或断力矩的明确降级状态。

推荐依次完成：离线 golden test、传感器回放、HIL、架空方向测试、固定机体 VMC、
系绳站立、低速直行、低速转向、高度小范围变化，最后才扩展到训练极限。第一阶段
建议使用训练 nominal VMC 增益，不要把 MuJoCo 中测试的更激进 PD 直接作为实机
默认值。
