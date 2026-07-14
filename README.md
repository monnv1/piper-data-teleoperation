# Piper 遥操作 + LeRobot 数据录制

## 1. 遥操作

`teleoperation/teleop.py`：主臂 `can1` 驱动从臂 `can0`。

```bash
conda run -n dynamicvla python3 -m teleoperation.teleop --speed-percent 5 --alpha 0.3
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--leader-can` | can1 | 主臂 CAN 接口 |
| `--follower-can` | can0 | 从臂 CAN 接口 |
| `--speed-percent` | 10 | MOVE J 速度百分比 |
| `--alpha` | 0.2 | 低通滤波系数 |
| `--max-step` | 2000 | 单周期最大关节变化，单位 0.001 度 |
| `--hz` | 50 | 控制频率 |
| `--offset` | [0,0,0,0,0,0] | 关节软件偏移，单位 0.001 度 |
| `--no-gripper` | false | 关闭夹爪跟随 |
| `--home` | false | Ctrl+C 后自动回到 `HOME_DEG` |

启动流程：

1. 确认两个 CAN 口都已 UP。
2. 两台机械臂上电。
3. 释放急停按钮。
4. 运行 `teleop.py`。
5. 拖动主臂，从臂跟随。

Ctrl+C 停止时，从臂默认保持最后位置；加 `--home` 会自动回零。

---

## 2. LeRobot 数据录制

`teleoperation/record_lerobot.py` 直接在遥操循环里调用 LeRobot 官方数据生成函数：

```python
LeRobotDataset.create(...)
dataset.add_frame(...)
dataset.save_episode()
```

不使用官方 `lerobot-record` 录制 CLI，也不需要先录 `.npz` 再转换。

本仓库当前依赖是 `lerobot==0.3.3`，本地包的 `CODEBASE_VERSION` 为 `v2.1`。脚本按当前本地 LeRobot API 生成数据集；如果升级到不同签名的 LeRobot v3 API，需要同步检查 `LeRobotDataset.create/add_frame/save_episode` 参数。

### 带双相机录制

```bash
conda run -n dynamicvla python3 -m teleoperation.record_lerobot \
  --repo-id local/piper-demo \
  --root ./data/piper-demo \
  --episodes 5 \
  --duration-s 30 \
  --fps 25 \
  --task "pick up the bottle" \
  --force
```

### 只验证机械臂和 parquet

```bash
conda run -n dynamicvla python3 -m teleoperation.record_lerobot \
  --repo-id local/piper-debug \
  --root ./data/piper-debug \
  --episodes 1 \
  --duration-s 10 \
  --fps 25 \
  --task "debug teleoperation" \
  --no-cameras \
  --force
```

默认读取 `deploy/configs/piper_gemini_d435i.yaml`：

- `observation.images.opst_cam`：RealSense D435i 固定第三视角。
- `observation.images.wrist_cam`：Orbbec Gemini 305 腕部视角。
- `observation.state`：从臂当前模型 TCP `[x, y, z, rx, ry, rz]`，单位为米和弧度。
- `action`：当前遥操目标的绝对模型 TCP + 夹爪开口 `[x, y, z, rx, ry, rz, gripper]`。

注意：录制文件里保存的是 **absolute action**。项目训练读取器在 `delta_action=True` 时会自动执行 `action[..., :6] -= observation.state[..., :6]`，不要在录制阶段提前保存 delta。

---

## 3. 数据集结构

```text
数据集根目录/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.opst_cam/
        │   └── episode_000000.mp4
        └── observation.images.wrist_cam/
            └── episode_000000.mp4
```

Features：

```python
features = {
    "observation.images.wrist_cam": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.opst_cam": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "rx", "ry", "rz"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
    },
}
```

每帧数据：

```python
frame = {
    "observation.images.wrist_cam": np.ndarray,  # H,W,3 RGB
    "observation.images.opst_cam": np.ndarray,   # H,W,3 RGB
    "observation.state": np.array([x, y, z, rx, ry, rz], dtype=np.float32),
    "action": np.array([x, y, z, rx, ry, rz, gripper], dtype=np.float32),
}
```

---

## 4. 训练

用项目自带训练流程：

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  -c configs/dynamicvla.yaml \
  -d local/piper-demo
```

也可以在 `configs/dynamicvla.yaml` 中调整 dataset 路径和 feature 列表。
