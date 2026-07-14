# Piper 遥操作 + 数据录制

双臂遥操作系统，leader（`can1`）驱动 follower（`can0`），支持三种录制模式直接生成 LeRobot 格式数据集。

## 目录结构

```
teleoperation/
├── teleop.py                  # 基础遥操作（纯跟随，不录制）
├── record_continuous.py       # 连续录制模式：常开跟随，ENTER 控制录制约
├── record_lerobot.py          # 定长录制模式：固定时长 episode，自动分段
├── record_interactive.py      # 示教录制模式：物理 teach 按钮，两段式归零
├── read_end_pose.py           # 只读末端位姿打印工具
├── record_j6.py               # J6 关节回绕诊断（CSV 数据记录）
├── see_zero_rotation.py       # IK 分析：将非零旋转轴置零后的求解结果
├── set_joint_zero.py          # 将当前物理位置设为关节硬件零位（校准工具）
├── piper_reference.md         # Piper SDK v0.6.1 中文参考手册（408 行）
└── __init__.py
```

## 硬件连接

| 角色 | CAN 接口 | 状态 |
|---|---|---|
| Leader（主臂，拖动端） | `can1` | Standby + 重力补偿；手动拖动 |
| Follower（从臂，执行端） | `can0` | MIT 高跟随模式（0xAD） |

CAN 速率必须为 1 Mbps：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

## 选择录制模式

| 录制模式 | 适用场景 | 操作方式 |
|---|---|---|
| `record_continuous.py` | 快速连续采集多段数据 | ENTER 开始/停止录制，leader 始终可拖动 |
| `record_lerobot.py` | 固定时长录制约，自动分段 | 按 ENTER 开始每段，自动停止后继续下一段 |
| `record_interactive.py` | 精确位置控制，避免归零碰撞 | 物理 teach 按钮拖动，两段式归零（J5 最后归） |

## 1. 纯遥操作（不录制）

```bash
python -m teleoperation.teleop --speed-percent 5 --alpha 0.3
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--leader-can` | can1 | 主臂 CAN |
| `--follower-can` | can0 | 从臂 CAN |
| `--speed-percent` | 10 | MOVE J 速度百分比 |
| `--alpha` | 0.2 | 低通滤波系数（越小越平滑） |
| `--max-step` | 2000 | 单周期最大关节变化（0.001° 单位） |
| `--hz` | 50 | 控制频率 |
| `--offset` | [0]*6 | 关节软件偏移（0.001° 单位） |
| `--no-gripper` | false | 关闭夹爪跟随 |
| `--home` | false | Ctrl+C 后自动回到 HOME_DEG |
| `--gripper-effort` | 300 | 夹爪跟随力度 |

## 2. 连续录制（推荐）

Leader 始终处于 Standby 可拖动状态，follower 全程 MIT 跟随。按 ENTER 开始/停止录制约。停止后立即异步转码视频，同时允许用户继续拖动 leader 至下一段起始位。

```bash
python -m teleoperation.record_continuous \
  --repo-id local/piper-demo \
  --root ./data/piper-demo \
  --fps 50 \
  --tasks tasks.json \
  --speed-percent 5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--leader-can` | can1 | |
| `--follower-can` | can0 | |
| `--repo-id` | local/piper-demo | 数据集标识 |
| `--root` | ./data | 数据集根目录 |
| `--fps` | 50 | 录制帧率 |
| `--tasks` | None | JSON 任务列表文件 |
| `--force` | false | 覆盖已有数据集 |
| `--disable-cameras` | false | 不录相机 |
| `--video-codec` | libx264 | 视频编码器 |
| `--speed-percent` | 10 | |
| `--alpha` | 0.3 | 低通滤波 |
| `--config` | deploy/configs/piper_gemini_d435i.yaml | 相机配置 |

## 3. 定长录制

固定时长 episode（默认 30s），follower 在录制期间 MIT 跟随、录制间隔 MOVE J 保持。录制结束后自动保存 episode，按 ENTER 开始下一段。

```bash
python -m teleoperation.record_lerobot \
  --repo-id local/piper-demo \
  --root ./data/piper-demo \
  --episodes 5 \
  --duration-s 30 \
  --fps 25 \
  --task "pick up the bottle" \
  --force
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--episodes` | 5 | 录制段数 |
| `--duration-s` | 30 | 每段时长 |
| `--fps` | 25 | 录制帧率 |
| `--start-immediately` | false | 跳过 ENTER 确认 |
| `--no-cameras` | false | 不录相机 |
| `--home` | false | 结束后回到 HOME_DEG |

## 4. 示教录制

Leader **不**进入 Standby，而是通过物理 teach 按钮进入拖动模式。录制时 follower 进入 MIT，停止后退出 MIT，两段式归零（先 J1-J4+J6，最后 J5）避免碰撞。适用于需要精确起始位置和避免归零碰撞的场景。

```bash
python -m teleoperation.record_interactive \
  --repo-id local/piper-demo \
  --root ./data/piper-demo \
  --task "pick up the bottle" \
  --repeat 5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--teach-off-timeout-s` | 30 | 等待 teach 按钮释放超时 |
| `--no-home-between` | false | 段间不回零 |
| `--home-deg` | [0,70.9,-60.9,0,58.4,0] | 归零关节角 |

## 坐标系

数据集使用模型 TCP 坐标系，与 `deploy` 一致：

- `observation.state`：`[x, y, z, rx, ry, rz]`（6 维），**从臂当前位姿**
- `action`：`[x, y, z, rx, ry, rz, gripper]`（7 维），**绝对位姿目标**

单位：位置为米，Euler 角为弧度，夹爪为米制开口量。

录制文件中保存 absolute action。训练读取器在 `delta_action=True` 时会自动执行 `action[..., :6] -= observation.state[..., :6]`。

## 数据集结构

```text
数据集根目录/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.opst_cam/
        │   └── episode_000000.mp4
        └── observation.images.wrist_cam/
            └── episode_000000.mp4
```

Features：

| Key | dtype | shape | 说明 |
|---|---|---|---|
| `observation.images.wrist_cam` | video | (480,640,3) | Gemini 305 腕部 |
| `observation.images.opst_cam` | video | (480,640,3) | D435i 第三视角 |
| `observation.state` | float32 | (6,) | 从臂模型 TCP |
| `action` | float32 | (7,) | 绝对位姿目标 |

## 诊断工具

| 工具 | 说明 |
|---|---|
| `read_end_pose.py` | 只读打印末端位姿 `[x,y,z,rx,ry,rz]`，`--can` `--hz` |
| `record_j6.py` | J6 回绕诊断：同时记录 SDK 关节角和原始电机位置，输出 CSV |
| `see_zero_rotation.py` | IK 分析：将非零旋转轴置零，观察求解结果和误差 |
| `set_joint_zero.py` | 将当前物理位置设为关节硬件零位，`--joint` `--repeat` |
| `piper_reference.md` | Piper SDK v0.6.1 完整中文手册（状态系统、控制命令、6 种运动模式） |

## 参考文档

`piper_reference.md` 包含 Piper SDK 详细参考：状态系统 10 种 arm_status、6 种运动模式（MOVE_P/J/L/C/M/CPV）、全部控制命令和读取函数、标准启动流程、遥操作架构说明（Leader Standby + 重力补偿；Follower MIT 高跟随）。

## 训练

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  -c configs/dynamicvla.yaml \
  -d local/piper-demo
```
