# Piper 机械臂 SDK 参考手册

> piper_sdk v0.6.1 | 安装于 `conda dynamicvla` 环境

---

## 一、状态系统

### 1.1 综合状态 `GetArmStatus()` — CAN ID: 0x2A1

```python
status = piper.GetArmStatus()
# 或直接读字段
status.ctrl_mode      # 控制模式
status.arm_status     # 机械臂状态
status.mode_feed      # 运动模式反馈
status.teach_status   # 示教状态
status.motion_status  # 运动状态
status.err_code       # 故障码 (16bit)
```

#### ctrl_mode — Byte 0：控制模式

| 值 | 枚举名 | 含义 |
|---|---|---|
| `0x00` | STANDBY | 待机模式（使能后只有重力补偿，无位置锁定） |
| `0x01` | CAN_CTRL | CAN指令控制模式（标准上位机控制） |
| `0x02` | TEACHING_MODE | 示教模式（只能物理按钮触发） |
| `0x03` | ETHERNET_CONTROL_MODE | 以太网控制模式 |
| `0x04` | WIFI_CONTROL_MODE | WiFi控制模式 |
| `0x05` | REMOTE_CONTROL_MODE | 遥控器控制模式 |
| `0x06` | LINKAGE_TEACHING_INPUT_MODE | 联动示教输入模式（主从硬联动） |
| `0x07` | OFFLINE_TRAJECTORY_MODE | 离线轨迹模式 |

#### arm_status — Byte 1：机械臂状态

| 值 | 含义 |
|---|---|
| `0x00` | 正常 |
| `0x01` | 急停 |
| `0x02` | 无解（逆解失败） |
| `0x03` | 奇异点 |
| `0x04` | 目标角度超限位 |
| `0x05` | 关节通信异常 |
| `0x06` | 关节抱闸未打开 |
| `0x07` | 碰撞发生 |
| `0x08` | 拖动示教时超速 |
| `0x09` | 关节状态异常 |
| `0x0A` | 其它异常 |
| `0x0B` | 示教记录中 |
| `0x0C` | 示教执行中 |
| `0x0D` | 示教暂停 |
| `0x0E` | 主控NTC过温 |
| `0x0F` | 释放电阻NTC过温 |

#### mode_feed — Byte 2：当前运动模式

| 值 | 含义 |
|---|---|
| `0x00` | MOVE_P — 点位运动（笛卡尔空间） |
| `0x01` | MOVE_J — 关节运动 |
| `0x02` | MOVE_L — 直线运动 |
| `0x03` | MOVE_C — 圆弧运动 |
| `0x04` | MOVE_M — MIT模式（V1.5-2+） |
| `0x05` | MOVE_CPV — 连续路径模式（V1.6.5+） |

#### teach_status — Byte 3：示教状态

| 值 | 含义 |
|---|---|
| `0x00` | 关闭 |
| `0x01` | 开始示教记录（进入拖动示教模式） |
| `0x02` | 结束示教记录（退出拖动示教模式） |
| `0x03` | 执行示教轨迹 |
| `0x04` | 暂停执行 |
| `0x05` | 继续执行 |
| `0x06` | 终止执行 |
| `0x07` | 运动到轨迹起点 |

#### motion_status — Byte 4：运动状态

| 值 | 含义 |
|---|---|
| `0x00` | 已到达目标位置 ✅ |
| `0x01` | 未到达目标位置（正在运动中） ⏳ |

#### err_code — Byte 6-7：故障码（16bit）

```
Byte 6 (角度超限位):
  bit[0]~bit[5] — 关节1~6 角度超限位 (1=异常)

Byte 7 (通信异常):  
  bit[0]~bit[5] — 关节1~6 通信异常 (1=异常)
```

---

### 1.2 使能状态 `GetArmEnableStatus()`

```python
enable_list = piper.GetArmEnableStatus()  # → [bool]*6
```

返回 6 个布尔值，对应关节 1~6 的 `driver_enable_status`，从低速反馈 CAN ID 0x261~0x266 读取。

---

## 二、控制指令

### 2.1 `MotionCtrl_1(emergency_stop, track_ctrl, grag_teach_ctrl)` — CAN ID: 0x150

```python
piper.MotionCtrl_1(
    emergency_stop: 0x00=无效 / 0x01=急停 / 0x02=恢复,
    track_ctrl:     轨迹指令,
    grag_teach_ctrl: 示教指令,
)
```

**emergency_stop：**

| 值 | 功能 |
|---|---|
| `0x00` | 无效 |
| `0x01` | ⛔ 快速急停 — 立即停止，机械臂缓慢下降 |
| `0x02` | 🔄 恢复 — 从急停/示教中恢复 |

**track_ctrl — 轨迹指令：**

| 值 | 功能 |
|---|---|
| `0x00` | 关闭 |
| `0x01` | 暂停当前规划 |
| `0x02` | 继续当前轨迹 ✅ 常用 |
| `0x03` | 清除当前轨迹 |
| `0x04` | 清除所有轨迹 |
| `0x05` | 获取当前规划轨迹 |
| `0x06` | 终止执行 |
| `0x07` | 轨迹传输 |
| `0x08` | 轨迹传输结束 |

**grag_teach_ctrl — 示教指令：**

| 值 | 功能 |
|---|---|
| `0x00` | 关闭 |
| `0x01` | ⚠️ 进入示教模式（**固件可能忽略**，只能用物理按钮） |
| `0x02` | ✅ 退出示教模式（需按钮松开后才有效） |
| `0x03` | 执行示教轨迹复现 |
| `0x04` | 暂停执行 |
| `0x05` | 继续执行 |
| `0x06` | 终止执行 |
| `0x07` | 运动到轨迹起点 |

### 2.2 `MotionCtrl_2(ctrl_mode, move_mode, speed, mit_mode)` — CAN ID: 0x151

```python
piper.MotionCtrl_2(
    ctrl_mode:   控制模式,
    move_mode:   运动模式,
    move_spd_rate_ctrl: 速度 0~100,
    is_mit_mode: MIT模式 0x00/0xAD/0xFF,
    residence_time: 0,
    installation_pos: 0x00,
)
```

**ctrl_mode：**

| 值 | 含义 |
|---|---|
| `0x00` | 待机模式（使能+重力补偿，无位置锁 ← 主臂遥操用） |
| `0x01` | CAN指令控制模式（标准控制 ← 从臂用） |
| `0x03` | 以太网控制模式 |
| `0x04` | WiFi控制模式 |
| `0x07` | 离线轨迹模式 |

**move_mode：**

| 值 | 含义 | 配套控制函数 |
|---|---|---|
| `0x00` | MOVE_P — 点位运动（弧线） | `EndPoseCtrl()` |
| `0x01` | **MOVE_J — 关节运动** | **`JointCtrl()`** |
| `0x02` | MOVE_L — 直线运动 | `EndPoseCtrl()` |
| `0x03` | MOVE_C — 圆弧运动 | `EndPoseCtrl()` + `MoveCAxisUpdateCtrl()` |
| `0x04` | MOVE_M — MIT模式 | `JointMitCtrl()` |
| `0x05` | MOVE_CPV | — |

**is_mit_mode：**

| 值 | 含义 | 特点 |
|---|---|---|
| `0x00` | **位置速度模式** | 有 S 曲线规划，平滑，精确，响应较慢 |
| `0xAD` | **MIT模式** | 跳过了轨迹规划，PD直驱电机，低延迟，响应快 |
| `0xFF` | 无效 | — |

> **注意**：`is_mit_mode` 控制的是底层电机控制算法，不是运动模式。  
> `move_mode=0x01 + mit_mode=0xAD` = 关节控制 + MIT 电机算（响应快）  
> `move_mode=0x04 + mit_mode=0xAD` = 真正的 MIT 控制模式（需配合 `JointMitCtrl`）

### 2.3 使能/失能 — CAN ID: 0x471

```python
piper.EnableArm(motor_num=7, enable_flag=0x02)  # 使能
piper.DisableArm(motor_num=7, enable_flag=0x01)  # 失能
piper.EnablePiper()    # → bool — 使能全部
piper.DisablePiper()   # → bool — 失能全部
```

**参数：**

| 参数 | 值 | 含义 |
|---|---|---|
| motor_num | `1~6` | 单个关节 |
| | `7` | 所有电机（含夹爪） |
| | `0xFF` | 全部关节电机 |
| enable_flag | `0x01` | 失能 |
| | `0x02` | 使能 |

---

## 三、运动控制函数

### 3.1 关节控制

```python
piper.JointCtrl(
    joint_1, joint_2, joint_3, joint_4, joint_5, joint_6
    # 单位: 0.001°
)
```

关节限位（单位：度）：

| 关节 | 下限 | 上限 |
|---|---|---|
| J1 | -150.0 | 150.0 |
| J2 | 0 | 180.0 |
| J3 | -170.0 | 0 |
| J4 | -100.0 | 100.0 |
| J5 | -70.0 | 70.0 |
| J6 | -120.0 | 120.0 |

单位换算：
```python
factor = 57295.7795  # 1000 * 180 / π
degree = 90          # 要发的角度
units = int(degree * factor)
```

### 3.2 末端笛卡尔控制

```python
piper.EndPoseCtrl(
    X, Y, Z,      # 单位: 0.001mm
    RX, RY, RZ,   # 单位: 0.001°
)

factor = 1000  # mm → 0.001mm
units = int(57.0 * factor)  # 57mm
```

### 3.3 夹爪控制

```python
piper.GripperCtrl(
    gripper_angle,    # 单位: 0.001mm
    gripper_effort,   # 力矩 0.001N·m, 范围 0~5000
    gripper_code,     # 0x00=关闭, 0x01=开启, 0x02=关闭清错, 0x03=开启清错
    set_zero,         # 0x00=无效, 0xAE=设为零位
)
```

### 3.4 圆弧控制

```python
piper.MotionCtrl_2(0x01, 0x03, speed, 0x00)  # 切 MOVE_C
piper.EndPoseCtrl(起点X, 起点Y, ...)
piper.MoveCAxisUpdateCtrl(0x01)  # 标记起点
piper.EndPoseCtrl(中点X, 中点Y, ...)
piper.MoveCAxisUpdateCtrl(0x02)  # 标记中点
piper.EndPoseCtrl(终点X, 终点Y, ...)
piper.MoveCAxisUpdateCtrl(0x03)  # 标记终点 → 开始走弧
```

### 3.5 MIT 控制

```python
piper.JointMitCtrl(
    motor_num,   # 关节号 1~6
    pos_ref,     # 目标位置 (rad), 范围 [-12.5, 12.5]
    vel_ref,     # 目标速度, 范围 [-45.0, 45.0]
    kp,          # 刚度（比例增益）, 参考 10, 范围 [0, 500]
    kd,          # 阻尼（微分增益）, 参考 0.8, 范围 [-5, 5]
    t_ref,       # 前馈力矩, 范围 [-18, 18] N·m
)
```

MIT 模式下实际力矩计算公式：
```
T_ref = kp * (pos_ref - θ) + kd * (vel_ref - dθ) + t_ff
```

---

## 四、读取函数

| 函数 | 返回值 | 单位 |
|---|---|---|
| `GetArmStatus()` | `ArmStatus{ctrl_mode, arm_status, mode_feed, teach_status, motion_status, err_code}` | 枚举 |
| `GetArmJointMsgs()` | `{joint_1~6}` | 0.001° |
| `GetArmEndPoseMsgs()` | `{X, Y, Z, RX, RY, RZ}` | 0.001mm / 0.001° |
| `GetArmGripperMsgs()` | `{grippers_angle, grippers_effort, foc_status}` | 0.001mm / 0.001N·m |
| `GetArmHighSpdInfoMsgs()` | `{motor_1~6: {motor_speed, current, pos, effort}}` | rad, A, N·m |
| `GetArmLowSpdInfoMsgs()` | `{motor_1~6: {vol, foc_temp, motor_temp, foc_status, bus_current}}` | 0.1V, ℃, 0.001A |
| `GetArmEnableStatus()` | `[bool]*6` | 6个关节使能状态 |
| `GetFK(mode)` | `6x6 list` 各关节正解 | 0.001mm / 0.001° |
| `GetCanFps()` | `float` | CAN帧率 |

---

## 五、速查：6种运动模式

| 模式 | 值 | 你发什么 | 路径形状 | 调用 |
|---|---|---|---|---|
| MOVE_P | `0x00` | 末端坐标 | 弧线 | `EndPoseCtrl(X,Y,Z,RX,RY,RZ)` |
| MOVE_J | `0x01` | 6个关节角度 | 各关节独立转 | `JointCtrl(j1..j6)` |
| MOVE_L | `0x02` | 末端坐标 | 直线 | `EndPoseCtrl(X,Y,Z,RX,RY,RZ)` |
| MOVE_C | `0x03` | 3组坐标 | 圆弧 | `EndPoseCtrl` + `MoveCAxisUpdateCtrl(1/2/3)` |
| MOVE_M | `0x04` | pos/vel/kp/kd/t_ref | 力矩/阻抗 | `JointMitCtrl(关节号, ...)` |
| MOVE_CPV | `0x05` | — | 连续路径 | — |

---

## 六、标准启动流程

```python
from piper_sdk import *

# 1. 连接
piper = C_PiperInterface_V2(can_name="can0",
                            start_sdk_joint_limit=True,
                            start_sdk_gripper_limit=True)
piper.ConnectPort()
time.sleep(0.5)

# 2. 配置从臂模式 (才能读到关节反馈)
piper.MasterSlaveConfig(0xFC, 0, 0, 0)
time.sleep(0.2)

# 3. 使能
while not piper.EnablePiper():
    time.sleep(0.01)

# 4. 控制
piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # CAN + MOVE_J + 30%速度
piper.JointCtrl(0, 0, 0, 0, 0, 0)         # 回零

# 5. 读取状态
status = piper.GetArmStatus()
print("运动状态:", status.motion_status)  # 0=到了, 1=没到
```

---

## 七、遥操典型架构

### 主臂 — 待机+重力补偿，可拖拽

```python
leader.EnablePiper()
leader.MotionCtrl_2(0x00, 0x01, 1, 0x00)   # ctrl_mode=0x00 Standby
```

### 从臂 — MIT 高跟随，低延迟

```python
follower.EnablePiper()
follower.MotionCtrl_2(0x01, 0x01, speed, 0xAD)  # mit_mode=0xAD
```

### 50Hz 跟随循环

```python
follower.MotionCtrl_2(0x01, 0x01, speed, 0xAD)
follower.JointCtrl(*target_joints)
```

---

## 八、注意事项

1. **示教模式只能物理按钮触发**，CAN 指令 `MotionCtrl_1(...,0x01)` 无效（GitHub Issue #11 确认）
2. **退出示教**可以通过 `MotionCtrl_1(0x00,0x00,0x02)`，但需物理按钮已松开
3. **MIT 模式(0xAD) 下 `clear trajectory` 会导致电机掉目标**，可能掉电
4. **MIT 模式(0xAD) 下发 `JointCtrl`** = 更新 PD 的 pos_ref，不是位置模式的位置锁
5. **主臂只能用 `ctrl_mode=0x00` Standby 模拟重力补偿**，不是真正的 TEACHING_MODE
6. **主从硬联动 (0x06)** 是硬件直连，不经过上位机，延迟最低但无法加滤波/限位

---

## 九、GitHub 资源

- 仓库: <https://github.com/agilexrobotics/piper_sdk>
- Issue #11 (示教模式软件触发): <https://github.com/agilexrobotics/piper_sdk/issues/11>
- SDK 接口文档: <https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD>
- Discord: <https://discord.gg/wrKYTxwDBd>
