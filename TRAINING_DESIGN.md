# 训练设计对齐记录

## 保留的 Fudan 设计

Actor 单帧严格为 25 维：机体角速度 3、投影重力 3、`vx/yaw/base-height` 指令 3、四个主动腿关节位置 4、六个执行关节速度 6、上一策略动作 6。五帧按时间顺序堆叠为 125 维，当前帧与 Encoder 估计的三维线速度拼成 28 维 Actor 输入。监督目标与 Fudan 一致，为真实机体系线速度乘观测尺度 2.0。

奖励函数与权重：

| term | weight |
|---|---:|
| tracking_lin_vel | 1.0 |
| tracking_lin_vel_enhance | 1.0 |
| tracking_ang_vel | 1.0 |
| tracking_ang_vel_enhance | 1.0 |
| base_height | 1.0 |
| nominal_state | -1.0 |
| lin_vel_z | -1.0 |
| ang_vel_xy | -0.20 |
| orientation | -100.0 |
| dof_vel | -5e-5 |
| dof_acc | -2.5e-7 |
| torques | -1e-4 |
| action_rate | -0.01 |
| action_smooth | -0.01 |
| collision | -1.0 |
| dof_pos_limits | -1.0 |

线速度和角速度跟踪使用 `exp(-e²/0.25)`；enhance 项使用 `exp(-e²/2.5)-1`；高度使用 `exp(-e²/0.001)`。每个加权项在乘以环境步长前限制到每秒 `[-1,1]`，总奖励不做 positive-only 裁剪。

## 必要的资产适配

Fudan 原仓库是开链四关节 position-PD；`infantry_2027_v0` 是具有四个球形闭环约束的五连杆机构，因此动作层必须换成经过 Isaac Sim 与 MuJoCo 双验证的 VMC。这个变化只发生在 policy action 到电机力矩的执行层，没有改变奖励集合。

左右 CAD 关节轴相反。VMC 先用 `left sign=+1/right sign=-1` 映射到统一几何坐标，计算 `Jᵀ[F,M]` 后再映射回物理关节。轮轴统一前进符号为 left `+1`、right `-1`。

Fudan 的 base-height 数值不能照搬。新资产髋关节原点相对 base_link 为 `z=0.07 m`，轮半径 `0.058 m`，直立几何得到 `base_height ≈ L-0.012 m`。所以 `L=0.16–0.33 m` 对应 `z=0.148–0.318 m`。

## 单次长训练而不是 DR 分阶段

全部 DR 维度从训练开始即存在。只有指令支持集在同一 optimizer、同一 run 内连续扩张；这不是多阶段 checkpoint 迁移。前 1500 次更新后固定为最终范围，余下 3500 次更新用于最终分布收敛。

转向语义与 Fudan 的实际平地实现一致，使用 direct yaw-rate。命令模式为 10% 静止站立（`vx=0,yaw=0`）、20% 原地转向（`vx=0,|yaw|>=0.5`）、70% 运动；运动样本要求 `|vx|>=0.15 m/s`。前 1500 次更新内运动 yaw 从 `±1` 连续扩展到 `±4 rad/s`，原地 yaw 从 `±2` 扩展到 `±10 rad/s`。

高度命令与 Fudan 一样在每次 5 s 重采样时瞬时跳变，不施加目标变化率。为了让一次从零训练覆盖完整行程，15% 高度样本直接取当前课程范围的上下端点，其余保持均匀采样。

## Terrain-v0 对齐记录

训练地形列严格使用参考最终分布：平地 0.50、正/负平滑坡各 0.10、正/负粗糙坡各 0.05、下/上楼梯各 0.10。单块 8 m×8 m，水平/垂直分辨率 0.1 m/0.005 m，10 行难度严格为 row/10，20 列按上述比例展开，初始最高等级 5。离散障碍、踏石、沟壑和深坑只留在审阅工具中，不参与 Terrain-v0。

Actor/Encoder 输入契约完全不变，保证部署时不需要地形传感器。Critic 比平地多 77 维局部高度扫描，总维数由 68 变为 145；高度奖励改为相对射线测得的局部地面，而不是世界坐标 z。地形课程每次 episode reset 检查：行进距离超过 4 m 升级；未升级且线速度跟踪 episode 平均低于 0.4 时降级。

Terrain-v0 从第 0 次更新启用参考的宽域随机化。每个环境在 episode reset 时固定分配为“水平推扰”或“向下冲击”之一，随后每 7 s 施加该类型的扰动；不会在同一 episode 内反复切换扰动类型。碰撞惩罚严格对齐参考最终训练，只统计 `base_link`、`lf_link`、`rf_link`，不把正常接地的轮子与全部连杆误算为碰撞。奖励采用参考最终 terrain 权重与逐项裁剪；执行层仍是闭链 VMC，因此动作维度保持 6。Terrain runner 每 20 次更新保存一次 checkpoint，降低长训练因外部进程或系统资源中断造成的回退量。

训练决策优先级：先从随机初始化做 200 updates 的短测；只要站立、episode length、跟踪奖励和 terrain level 呈可学习趋势，就从零启动正式长训练。仅当短测明确失败时，才增加“只加载平地 actor+Encoder 参数”的可选初始化路径，critic、optimizer 和 iteration 均重新开始。
