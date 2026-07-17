"""连续遥操作录制 — 主臂持续可拖拽，从臂持续跟随，ENTER 控制录与不录。

流程：
  1. 连接双臂 → 使能 → 主臂 Standby（重力补偿，可自由拖拽）
                            从臂 MIT 高跟随模式（循环跟随主臂）
  2. 固定频率跟随循环持续运行，从臂一直跟着主臂走
  3. ENTER → 开始录制数据帧
  4. ENTER → 停止录制 → 保存该段数据（后处理期间双臂照常跟随）
  5. 用户趁后处理时布置场地，拖主臂到新位置
  6. 重复 3-5 直到全部 task 完成
  7. 锁住关节，断开 CAN，退出

关键特性：
  - 全程不切换模式：主臂 Standby(0x00)，从臂 MIT(0xAD)
  - 无归位、无控制权争夺、无示教按钮等待
  - 后处理与跟随并行，不阻塞用户操作
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deploy.common.latest import FrameBuffer
from deploy.config import load_config
from deploy.devices.factory import create_camera
from deploy.devices.piper_frames import PiperFrameTransform
from teleoperation.teleop import (
    ALPHA,
    FOLLOWER_CAN,
    GRIPPER_ALPHA,
    GRIPPER_MAX_STEP_UNITS,
    JOINT_OFFSET,
    LEADER_CAN,
    MAX_STEP_UNITS,
    make_piper,
    map_follower_target,
    read_gripper,
    read_joints,
)

DEFAULT_DEPLOY_CONFIG = Path("deploy/configs/piper_gemini_d435i.yaml")


# ── helpers ─────────────────────────────────────────────────────────────

def _read_sdk_pose_units(piper) -> np.ndarray:
    pose = piper.GetArmEndPoseMsgs().end_pose
    return np.asarray(
        [pose.X_axis, pose.Y_axis, pose.Z_axis,
         pose.RX_axis, pose.RY_axis, pose.RZ_axis],
        dtype=np.float64,
    )


def _model_pose_from_sdk_units(pose_units: np.ndarray,
                               frames: PiperFrameTransform) -> np.ndarray:
    sdk_position_m = pose_units[:3] / 1_000_000.0
    sdk_euler_rad = pose_units[3:] * (math.pi / 180_000.0)
    position_m, euler_rad = frames.sdk_to_model_pose(sdk_position_m, sdk_euler_rad)
    return np.concatenate([position_m.astype(np.float32), euler_rad.astype(np.float32)])


def _model_pose_from_target_joints(fk, target_joint_units: list[int],
                                   frames: PiperFrameTransform) -> np.ndarray:
    joint_radians = np.radians(np.asarray(target_joint_units, dtype=np.float64) / 1000.0)
    sdk_pose = np.asarray(fk.CalFK(joint_radians)[-1], dtype=np.float64)
    sdk_position_m = sdk_pose[:3] / 1000.0
    sdk_euler_rad = np.radians(sdk_pose[3:])
    position_m, euler_rad = frames.sdk_to_model_pose(sdk_position_m, sdk_euler_rad)
    return np.concatenate([position_m.astype(np.float32), euler_rad.astype(np.float32)])


def _read_gripper_m(piper) -> float | None:
    value = read_gripper(piper)
    # A missing follower measurement must never be turned into a valid-looking
    # closed-gripper observation. LeRobot records Present_Position as the
    # observation; if Piper cannot provide it, the whole sample is invalid.
    return None if value is None else float(value) / 1_000_000.0


# ── dataset ─────────────────────────────────────────────────────────────

def _build_features(config, use_cameras: bool) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "observation.leader_joints": {
            "dtype": "float32", "shape": (6,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
        "observation.follower_joints": {
            "dtype": "float32", "shape": (6,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        # LeRobot's built-in timestamp stays on its nominal frame_index / fps
        # grid. These fields preserve the actual acquisition/send times.
        "recording.sample_monotonic_ns": {
            "dtype": "int64", "shape": (1,), "names": ["nanoseconds"],
        },
        "recording.sample_wall_time_ns": {
            "dtype": "int64", "shape": (1,), "names": ["unix_nanoseconds"],
        },
        "recording.observation_monotonic_ns": {
            "dtype": "int64", "shape": (1,), "names": ["nanoseconds"],
        },
        "recording.action_sent_monotonic_ns": {
            "dtype": "int64", "shape": (1,), "names": ["nanoseconds"],
        },
        "recording.actual_dt_s": {
            "dtype": "float32", "shape": (1,), "names": ["seconds"],
        },
        "recording.actual_fps": {
            "dtype": "float32", "shape": (1,), "names": ["hertz"],
        },
        "recording.target_fps": {
            "dtype": "float32", "shape": (1,), "names": ["hertz"],
        },
    }
    if use_cameras:
        for name, cam_cfg in config.cameras.items():
            if not cam_cfg.enabled:
                continue
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (cam_cfg.height, cam_cfg.width, 3),
                "names": ["height", "width", "channel"],
            }
            features[f"recording.camera_host_timestamp_ns.{name}"] = {
                "dtype": "int64", "shape": (1,), "names": ["nanoseconds"],
            }
            features[f"recording.camera_device_timestamp_ms.{name}"] = {
                "dtype": "float64", "shape": (1,), "names": ["milliseconds"],
            }
            features[f"recording.camera_frame_number.{name}"] = {
                "dtype": "int64", "shape": (1,), "names": ["frame_number"],
            }
            features[f"recording.camera_age_ms.{name}"] = {
                "dtype": "float32", "shape": (1,), "names": ["milliseconds"],
            }
    return features


def _configure_lerobot_video_codec(video_codec: str) -> None:
    if video_codec == "libsvtav1":
        return
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    import lerobot.datasets.video_utils as video_utils
    original_encode = video_utils.encode_video_frames

    def encode_with_codec(imgs_dir, video_path, fps, **kwargs):
        kwargs.setdefault("vcodec", video_codec)
        return original_encode(imgs_dir, video_path, fps, **kwargs)

    video_utils.encode_video_frames = encode_with_codec
    lerobot_dataset_module.encode_video_frames = encode_with_codec


def _create_dataset(args, config, use_cameras: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _configure_lerobot_video_codec(args.video_codec)
    root = Path(args.root).expanduser() if args.root else None
    tasks_data: str | None = None
    if root is not None and root.exists():
        if not args.force:
            raise FileExistsError(
                f"Dataset root already exists: {root}. Re-run with --force."
            )
        tasks_path = root / "tasks.json"
        if tasks_path.exists():
            tasks_data = tasks_path.read_text(encoding="utf-8")
        shutil.rmtree(root)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="piper",
        features=_build_features(config, use_cameras),
        use_videos=use_cameras,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    # LeRobot 建完根目录后，恢复 tasks.json
    if tasks_data is not None:
        (root / "tasks.json").write_text(tasks_data, encoding="utf-8")
    return dataset


# ── cameras ─────────────────────────────────────────────────────────────

def _start_cameras(config, enabled: bool) -> tuple[dict[str, FrameBuffer], list]:
    if not enabled:
        return {}, []
    buffers: dict[str, FrameBuffer] = {}
    cams = []
    for name, cam_cfg in config.cameras.items():
        if not cam_cfg.enabled:
            continue
        buf = FrameBuffer(maxlen=max(120, int(cam_cfg.fps * 5)))
        cam = create_camera(name, cam_cfg, buf)
        cam.start()
        buffers[name] = buf
        cams.append(cam)
    return buffers, cams


def _wait_for_cameras(buffers: dict[str, FrameBuffer], timeout_s: float) -> None:
    if not buffers:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(b.latest() is not None for b in buffers.values()):
            return
        time.sleep(0.05)
    missing = [n for n, b in buffers.items() if b.latest() is None]
    raise TimeoutError(f"Timed out waiting for camera frames: {missing}")


def _camera_observation(
    buffers: dict[str, FrameBuffer],
    observation_timestamp_ns: int,
    tolerance_ms: float,
) -> dict[str, np.ndarray] | None:
    """Snapshot every latest camera frame as part of the observation.

    This mirrors LeRobot's robot.get_observation(): camera data belongs to the
    observation and is captured before reading/sending the teleoperator action.
    A partial camera observation is never written.
    """
    tolerance_ns = int(tolerance_ms * 1_000_000.0)
    observation = {}
    for name, buf in buffers.items():
        frame = buf.latest()
        if frame is None:
            logging.warning("No frame available for %s; skipping sample", name)
            return None
        age_ns = observation_timestamp_ns - frame.host_timestamp_ns
        if abs(age_ns) > tolerance_ns:
            logging.warning(
                "Latest %s frame is %.1f ms from observation (limit %.1f ms); "
                "skipping sample",
                name,
                age_ns / 1_000_000.0,
                tolerance_ms,
            )
            return None
        observation[f"observation.images.{name}"] = np.asarray(frame.rgb).copy()
        observation[f"recording.camera_host_timestamp_ns.{name}"] = np.asarray(
            [frame.host_timestamp_ns], dtype=np.int64
        )
        observation[f"recording.camera_device_timestamp_ms.{name}"] = np.asarray(
            [frame.device_timestamp_ms], dtype=np.float64
        )
        observation[f"recording.camera_frame_number.{name}"] = np.asarray(
            [frame.frame_number], dtype=np.int64
        )
        observation[f"recording.camera_age_ms.{name}"] = np.asarray(
            [age_ns / 1_000_000.0], dtype=np.float32
        )
    return observation


# ── arm status helpers (minimal, for logging only) ──────────────────────

def _format_arm_state(piper) -> str:
    try:
        wrapper = piper.GetArmStatus()
        status = getattr(wrapper, "arm_status", wrapper)
        ctrl = int(getattr(status, "ctrl_mode", -1))
        arm = int(getattr(status, "arm_status", -1))
        enable = piper.GetArmEnableStatus()
        enabled = "".join("1" if x else "0" for x in enable) if enable else "?"
        return f"ctrl=0x{ctrl:02X} arm=0x{arm:02X} enable={enabled}"
    except Exception:
        return "unknown"


# ── arm lifecycle ───────────────────────────────────────────────────────
# Leader:  Standby(0x00) + 使能 = 重力补偿，自由拖拽
# Follower: MIT(0xAD) 高跟随模式，持续跟随主臂
# 全程不切换模式，不归位，不争夺控制权


def _enable_until_ok(piper, label: str, timeout_s: float = 10.0) -> None:
    """使能直到所有电机正常，用于启动阶段。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        piper.EnablePiper()
        time.sleep(0.2)
        try:
            enable = piper.GetArmEnableStatus()
            if enable and all(enable):
                return
        except Exception:
            pass
    raise TimeoutError(f"{label}: motors did not enable within {timeout_s:.1f}s")


def _lock_and_disconnect(leader, follower, *,
                         leader_connected: bool,
                         follower_connected: bool,
                         command_gripper: bool,
                         gripper_effort: int) -> None:
    """锁住关节，断开 CAN。"""
    try:
        if follower_connected:
            try:
                joints = read_joints(follower)
                follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
                follower.JointCtrl(*joints)
                if command_gripper:
                    g = read_gripper(follower)
                    if g is not None:
                        follower.GripperCtrl(g, gripper_effort, 0x01, 0)
                time.sleep(0.2)
            except Exception as e:
                logging.warning("Could not lock follower: %s", e)
        if leader_connected:
            try:
                joints = read_joints(leader)
                leader.MotionCtrl_2(0x01, 0x01, 1, 0x00)
                leader.JointCtrl(*joints)
                time.sleep(0.1)
            except Exception as e:
                logging.warning("Could not lock leader: %s", e)
    finally:
        if follower_connected:
            try:
                follower.DisconnectPort()
            except Exception as e:
                logging.warning("Could not disconnect follower: %s", e)
        if leader_connected:
            try:
                leader.DisconnectPort()
            except Exception as e:
                logging.warning("Could not disconnect leader: %s", e)


# ── recording ───────────────────────────────────────────────────────────

def _keyboard_listener(stop_event: threading.Event) -> None:
    """ENTER 按键监听线程。"""
    try:
        input()
        stop_event.set()
    except EOFError:
        pass


class _FollowState:
    """跟随线程和主线程之间共享的状态。"""
    def __init__(self):
        self.last_filtered: list[int] | None = None
        self.last_gripper_units: int | None = None
        self.current_task: str | None = None
        self.recording: bool = False
        self.stop_event = threading.Event()
        self.record_lock = threading.Lock()
        self.recorded_sample_ns: list[int] = []
        self.skipped_camera_samples: int = 0
        self.skipped_gripper_samples: int = 0
        self.control_overruns: int = 0

    def begin_episode(self, task: str) -> None:
        with self.record_lock:
            self.current_task = task
            self.recorded_sample_ns = []
            self.skipped_camera_samples = 0
            self.skipped_gripper_samples = 0
            self.control_overruns = 0
            self.recording = True

    def finish_episode(self) -> tuple[list[int], int, int, int]:
        with self.record_lock:
            self.recording = False
            self.current_task = None
            return (
                self.recorded_sample_ns.copy(),
                self.skipped_camera_samples,
                self.skipped_gripper_samples,
                self.control_overruns,
            )


@dataclass(frozen=True)
class _EpisodeTimingSummary:
    frames: int
    target_fps: float
    measured_fps: float
    duration_s: float
    mean_dt_s: float
    median_dt_s: float
    min_dt_s: float
    max_dt_s: float
    skipped_camera_samples: int
    skipped_gripper_samples: int
    control_overruns: int


def _summarize_episode_timing(
    timestamps_ns: list[int],
    target_fps: float,
    skipped_camera_samples: int,
    skipped_gripper_samples: int,
    control_overruns: int,
) -> _EpisodeTimingSummary:
    if len(timestamps_ns) < 2:
        return _EpisodeTimingSummary(
            frames=len(timestamps_ns),
            target_fps=float(target_fps),
            measured_fps=0.0,
            duration_s=0.0,
            mean_dt_s=0.0,
            median_dt_s=0.0,
            min_dt_s=0.0,
            max_dt_s=0.0,
            skipped_camera_samples=skipped_camera_samples,
            skipped_gripper_samples=skipped_gripper_samples,
            control_overruns=control_overruns,
        )
    deltas_s = np.diff(np.asarray(timestamps_ns, dtype=np.int64)) / 1e9
    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    return _EpisodeTimingSummary(
        frames=len(timestamps_ns),
        target_fps=float(target_fps),
        measured_fps=(len(timestamps_ns) - 1) / duration_s,
        duration_s=duration_s,
        mean_dt_s=float(np.mean(deltas_s)),
        median_dt_s=float(np.median(deltas_s)),
        min_dt_s=float(np.min(deltas_s)),
        max_dt_s=float(np.max(deltas_s)),
        skipped_camera_samples=skipped_camera_samples,
        skipped_gripper_samples=skipped_gripper_samples,
        control_overruns=control_overruns,
    )


def _append_episode_timing_metadata(
    dataset_root: Path,
    episode_index: int,
    task: str,
    summary: _EpisodeTimingSummary,
) -> None:
    metadata_path = Path(dataset_root) / "meta" / "recording_timing.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_index": episode_index,
        "task": task,
        **summary.__dict__,
        "timestamp_semantics": {
            "timestamp": "LeRobot nominal frame_index / fps seconds",
            "recording.sample_monotonic_ns": "actual host monotonic sample start",
            "recording.sample_wall_time_ns": "actual Unix wall-clock sample start",
            "recording.observation_monotonic_ns": "follower observation complete",
            "recording.action_sent_monotonic_ns": "follower command sent",
        },
    }
    with metadata_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sleep_until_ns(deadline_ns: int) -> None:
    """Sleep most of the interval, then yield-spin to the monotonic deadline."""
    while True:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 1_000_000:
            time.sleep((remaining_ns - 500_000) / 1e9)
        else:
            time.sleep(0)


def _follow_loop(
    leader,
    follower,
    state: _FollowState,
    dataset,
    fps: int,
    speed_percent: int,
    alpha: float,
    max_step_units: int,
    joint_offset: list[int],
    command_gripper: bool,
    gripper_effort: int,
    gripper_alpha: float,
    gripper_max_step_units: int,
    camera_buffers: dict[str, FrameBuffer],
    camera_tolerance_ms: float,
    fk,
    frames: PiperFrameTransform,
) -> None:
    """LeRobot-style loop: observation -> teleop action -> send -> record."""
    period_ns = int(round(1_000_000_000 / fps))
    next_deadline_ns = time.monotonic_ns()

    while not state.stop_event.is_set():
        sample_start_ns = time.monotonic_ns()
        sample_wall_time_ns = time.time_ns()
        with state.record_lock:
            capture_recording_frame = state.recording

        # ── 1. observation：先完整读取从臂反馈和相机 ──
        follower_raw = read_joints(follower)
        follower_joints_deg = (
            np.asarray(follower_raw, dtype=np.float32) / 1000.0
        ).astype(np.float32)
        state_pose = _model_pose_from_sdk_units(
            _read_sdk_pose_units(follower), frames
        )
        follower_gripper_m = _read_gripper_m(follower)
        state_pose = (
            None
            if follower_gripper_m is None
            else np.concatenate(
                [
                    state_pose,
                    np.asarray([follower_gripper_m], dtype=np.float32),
                ]
            ).astype(np.float32)
        )
        observation_timestamp_ns = time.monotonic_ns()
        camera_observation = (
            _camera_observation(
                camera_buffers,
                observation_timestamp_ns,
                camera_tolerance_ms,
            )
            if capture_recording_frame and camera_buffers
            else {}
        )

        # ── 2. teleop action：再读取主臂并生成实际发送目标 ──
        raw = read_joints(leader)
        leader_joints_deg = (
            np.asarray(raw, dtype=np.float32) / 1000.0
        ).astype(np.float32)

        # ── 低通 + 限速 ──
        if state.last_filtered is None:
            state.last_filtered = raw[:]
        filtered = []
        for i in range(6):
            val = int(state.last_filtered[i] * (1.0 - alpha) + raw[i] * alpha)
            delta = val - state.last_filtered[i]
            delta = max(-max_step_units, min(max_step_units, delta))
            filtered.append(state.last_filtered[i] + delta)
        state.last_filtered = filtered
        target_joints = map_follower_target(filtered, joint_offset)

        # ── 夹爪 ──
        gripper_units = read_gripper(leader)
        if gripper_units is not None:
            if state.last_gripper_units is None:
                state.last_gripper_units = gripper_units
            fg = int(state.last_gripper_units * (1.0 - gripper_alpha) + gripper_units * gripper_alpha)
            dg = fg - state.last_gripper_units
            dg = max(-gripper_max_step_units, min(gripper_max_step_units, dg))
            state.last_gripper_units = state.last_gripper_units + dg
            if command_gripper:
                follower.GripperCtrl(state.last_gripper_units, gripper_effort, 0x01, 0)

        # ── 3. send_action：发送经过滤波/限速后的实际目标 ──
        follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
        follower.JointCtrl(*target_joints)
        action_sent_timestamp_ns = time.monotonic_ns()

        # ── 4. record：保存 observation + 实际发送的 action ──
        if capture_recording_frame:
            with state.record_lock:
                if state.recording:
                    task = state.current_task
                    if task is None:
                        logging.error(
                            "Recording is active without a task; skipping sample"
                        )
                    elif state_pose is None:
                        # Keep following the leader, but never save a sample
                        # whose follower gripper observation is fabricated.
                        state.skipped_gripper_samples += 1
                    elif camera_observation is None:
                        state.skipped_camera_samples += 1
                    else:
                        previous_ns = (
                            state.recorded_sample_ns[-1]
                            if state.recorded_sample_ns
                            else None
                        )
                        actual_dt_s = (
                            0.0
                            if previous_ns is None
                            else (sample_start_ns - previous_ns) / 1e9
                        )
                        actual_fps = (
                            0.0 if actual_dt_s <= 0 else 1.0 / actual_dt_s
                        )
                        action_gripper_m = (
                            float(state.last_gripper_units) / 1_000_000.0
                            if state.last_gripper_units is not None
                            else float(follower_gripper_m)
                        )
                        action_pose = _model_pose_from_target_joints(
                            fk, target_joints, frames
                        )
                        frame = {
                            "observation.state": state_pose,
                            "observation.leader_joints": leader_joints_deg,
                            "observation.follower_joints": follower_joints_deg,
                            "action": np.concatenate(
                                [
                                    action_pose,
                                    np.asarray(
                                        [action_gripper_m], dtype=np.float32
                                    ),
                                ]
                            ).astype(np.float32),
                            "recording.sample_monotonic_ns": np.asarray(
                                [sample_start_ns], dtype=np.int64
                            ),
                            "recording.sample_wall_time_ns": np.asarray(
                                [sample_wall_time_ns], dtype=np.int64
                            ),
                            "recording.observation_monotonic_ns": np.asarray(
                                [observation_timestamp_ns], dtype=np.int64
                            ),
                            "recording.action_sent_monotonic_ns": np.asarray(
                                [action_sent_timestamp_ns], dtype=np.int64
                            ),
                            "recording.actual_dt_s": np.asarray(
                                [actual_dt_s], dtype=np.float32
                            ),
                            "recording.actual_fps": np.asarray(
                                [actual_fps], dtype=np.float32
                            ),
                            "recording.target_fps": np.asarray(
                                [fps], dtype=np.float32
                            ),
                        }
                        frame.update(camera_observation)
                        # Omit timestamp= intentionally: official LeRobot
                        # keeps timestamp on frame_index / dataset.fps.
                        dataset.add_frame(frame, task=task)
                        state.recorded_sample_ns.append(sample_start_ns)

        # ── fixed-rate control，扣除本轮处理耗时 ──
        next_deadline_ns += period_ns
        now_ns = time.monotonic_ns()
        if now_ns <= next_deadline_ns:
            _sleep_until_ns(next_deadline_ns)
        else:
            missed_intervals = (now_ns - next_deadline_ns) // period_ns + 1
            with state.record_lock:
                if state.recording:
                    state.control_overruns += int(missed_intervals)
            logging.warning(
                "Control loop overrun by %.2f ms (%d interval%s)",
                (now_ns - next_deadline_ns) / 1_000_000.0,
                missed_intervals,
                "s" if missed_intervals != 1 else "",
            )
            # Do not replay missed control points in a burst.
            next_deadline_ns = now_ns


def run_continuous_record(
    *,
    dataset,
    leader,
    follower,
    fk,
    frames: PiperFrameTransform,
    tasks: list[str],
    fps: int,
    speed_percent: int,
    alpha: float,
    max_step_units: int,
    joint_offset: list[int],
    command_gripper: bool,
    gripper_effort: int,
    gripper_alpha: float,
    gripper_max_step_units: int,
    camera_buffers: dict[str, FrameBuffer],
    camera_tolerance_ms: float,
) -> None:
    """主循环：固定频率持续跟随，主线程 ENTER 控制录制启停。"""
    state = _FollowState()

    # 首次使能夹爪
    if command_gripper:
        g0 = read_gripper(leader)
        if g0 is not None:
            follower.GripperCtrl(g0, gripper_effort, 0x02, 0)
            time.sleep(0.05)
            follower.GripperCtrl(g0, gripper_effort, 0x03, 0)
            time.sleep(0.15)
            logging.info("Gripper enabled from leader: %.1f mm", g0 / 1000.0)
        else:
            fallback = read_gripper(follower)
            if fallback is not None:
                follower.GripperCtrl(fallback, gripper_effort, 0x02, 0)
                time.sleep(0.05)
                follower.GripperCtrl(fallback, gripper_effort, 0x03, 0)
                time.sleep(0.15)

    # 从臂进入 MIT 跟随模式
    follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)

    # 启动后台跟随线程
    follow_thread = threading.Thread(
        target=_follow_loop,
        args=(leader, follower, state, dataset, fps,
              speed_percent, alpha, max_step_units, joint_offset,
              command_gripper, gripper_effort, gripper_alpha,
              gripper_max_step_units, camera_buffers, camera_tolerance_ms,
              fk, frames),
        daemon=True,
    )
    follow_thread.start()
    logging.info("Follow loop started — follower tracks leader continuously.")

    for ep_idx, task in enumerate(tasks):
        print(f"\n{'─'*60}")
        print(f"  Episode {ep_idx + 1}/{len(tasks)}")
        print(f"  Task: {task}")
        print(f"{'─'*60}")
        input("  Press ENTER to START recording …")

        dataset.episode_buffer = dataset.create_episode_buffer()
        state.begin_episode(task)
        logging.info("Recording — press ENTER to stop")

        input()
        (
            timestamps_ns,
            skipped_camera_samples,
            skipped_gripper_samples,
            control_overruns,
        ) = (
            state.finish_episode()
        )
        if not timestamps_ns:
            raise RuntimeError(
                "Episode contains no valid frames. Check camera freshness and "
                "increase --camera-sync-tolerance-ms only if timestamps justify it."
            )

        logging.info("Stopped — saving episode …")
        dataset.save_episode()
        timing_summary = _summarize_episode_timing(
            timestamps_ns,
            fps,
            skipped_camera_samples,
            skipped_gripper_samples,
            control_overruns,
        )
        _append_episode_timing_metadata(
            dataset.root,
            ep_idx,
            task,
            timing_summary,
        )
        logging.info("Episode saved: %s", task)
        logging.info(
            "Timing: target=%.3f Hz measured=%.3f Hz frames=%d "
            "median_dt=%.3f ms skipped_camera=%d skipped_gripper=%d "
            "overruns=%d",
            timing_summary.target_fps,
            timing_summary.measured_fps,
            timing_summary.frames,
            timing_summary.median_dt_s * 1000.0,
            timing_summary.skipped_camera_samples,
            timing_summary.skipped_gripper_samples,
            timing_summary.control_overruns,
        )
        if (
            timing_summary.measured_fps > 0
            and abs(timing_summary.measured_fps - timing_summary.target_fps)
            / timing_summary.target_fps
            > 0.05
        ):
            logging.warning(
                "Measured episode rate differs from the LeRobot nominal fps by "
                "more than 5%%. Use recording_timing.jsonl as ground truth and "
                "lower --fps before collecting the final dataset."
            )

    state.stop_event.set()
    follow_thread.join(timeout=2.0)
    logging.info("All %d episodes complete — locking and disconnecting.", len(tasks))


# ── main ────────────────────────────────────────────────────────────────

def _load_tasks(tasks_file: str) -> list[str]:
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected tasks format in {tasks_file}: expected list or dict with 'tasks' key"
        )
    if not data:
        raise ValueError(f"No tasks found in {tasks_file}")
    if not all(isinstance(task, str) and task.strip() for task in data):
        raise ValueError(f"Every task in {tasks_file} must be a non-empty string")
    return [task.strip() for task in data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous teleop recording — leader always drag, follower always follow"
    )
    # ── arms ──
    parser.add_argument("--leader-can", default=LEADER_CAN)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--speed-percent", type=int, default=10)

    # ── filter ──
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-step", type=int, default=MAX_STEP_UNITS)
    parser.add_argument("--gripper-alpha", type=float, default=GRIPPER_ALPHA)
    parser.add_argument("--gripper-max-step", type=int, default=GRIPPER_MAX_STEP_UNITS)
    parser.add_argument("--offset", type=int, nargs=6, default=JOINT_OFFSET)

    # ── dataset ──
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Path to tasks JSON file (list of task strings)")
    parser.add_argument("--video-codec", type=str, default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=4)

    # ── config ──
    parser.add_argument("--config", type=str, default=str(DEFAULT_DEPLOY_CONFIG))

    # ── cameras ──
    parser.add_argument("--disable-cameras", action="store_true",
                        help="Disable video recording (default: enabled)")
    parser.add_argument("--camera-startup-timeout-s", type=float, default=10.0)
    parser.add_argument("--camera-sync-tolerance-ms", type=float, default=50.0)

    # ── home ──
    parser.add_argument("--home-deg", type=float, nargs=6,
                        default=[0.0, 90.0, -90.0, 0.0, 69.0, 0.0],
                        help="Starting home joint degrees (only used for initial pose)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be in [1, 100]")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.camera_sync_tolerance_ms <= 0:
        raise ValueError("--camera-sync-tolerance-ms must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if not 0.0 <= args.gripper_alpha <= 1.0:
        raise ValueError("--gripper-alpha must be in [0, 1]")

    # ── load config & tasks ──
    config = load_config(args.config)
    tasks = _load_tasks(args.tasks)
    logging.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    # ── kinematics & frames ──
    from piper_sdk.kinematics import C_PiperForwardKinematics
    fk = C_PiperForwardKinematics(config.robot.dh_is_offset)
    frames = PiperFrameTransform(
        config.robot.sdk_to_model_translation_m,
        config.robot.sdk_to_model_euler_xyz_rad,
    )

    # ── cameras ──
    use_cameras = (
        not args.disable_cameras
        and hasattr(config, "cameras")
        and any(c.enabled for c in config.cameras.values())
    )
    if use_cameras:
        camera_rates = {
            name: camera.fps
            for name, camera in config.cameras.items()
            if camera.enabled
        }
        slower_cameras = {
            name: rate for name, rate in camera_rates.items() if rate < args.fps
        }
        if slower_cameras:
            logging.warning(
                "Requested control/data rate is %d Hz but cameras are slower: %s. "
                "Frames may repeat; per-frame camera timestamps and frame numbers "
                "will preserve the true image timing.",
                args.fps,
                slower_cameras,
            )
    logging.info(
        "Recording rate target: %d Hz (%.3f ms period); actual timing is stored "
        "per frame and in meta/recording_timing.jsonl",
        args.fps,
        1000.0 / args.fps,
    )
    camera_buffers, cameras = _start_cameras(config, use_cameras)

    # ── arms ──
    leader = make_piper(args.leader_can)
    follower = make_piper(args.follower_can)
    leader_connected = False
    follower_connected = False
    command_gripper = not args.no_gripper
    dataset = None

    try:
        # ── connect ──
        logging.info("Connecting leader on %s", args.leader_can)
        leader.ConnectPort(piper_init=False, start_thread=True)
        leader_connected = True
        time.sleep(0.5)

        logging.info("Connecting follower on %s", args.follower_can)
        follower.ConnectPort(piper_init=False, start_thread=True)
        follower_connected = True
        time.sleep(0.5)

        # ── configure as slave mode ──
        leader.MasterSlaveConfig(0xFC, 0, 0, 0)
        follower.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.2)

        # ── 先使能从臂锁住，不让它耷拉着 ──
        logging.info("Enabling follower …")
        _enable_until_ok(follower, "follower")
        time.sleep(0.2)
        home_joints = read_joints(follower)
        logging.info("Initial follower joints [deg]: %s",
                     [round(j / 1000.0, 2) for j in home_joints])
        follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
        follower.JointCtrl(*home_joints)
        time.sleep(0.3)

        # ── 再使能主臂 ──
        logging.info("Enabling leader (Standby + gravity compensation) …")
        _enable_until_ok(leader, "leader")
        leader.MotionCtrl_2(0x00, 0x01, 1, 0x00)   # Standby: 重力补偿可拖
        time.sleep(0.3)

        # ── cameras（可能会慢，但此时从臂已经锁住了） ──
        if use_cameras:
            logging.info("Waiting for camera frames …")
            _wait_for_cameras(camera_buffers, args.camera_startup_timeout_s)

        # ── create dataset ──
        dataset = _create_dataset(args, config, use_cameras)

        # ── 进入主循环 ──
        run_continuous_record(
            dataset=dataset,
            leader=leader,
            follower=follower,
            fk=fk,
            frames=frames,
            tasks=tasks,
            fps=args.fps,
            speed_percent=args.speed_percent,
            alpha=args.alpha,
            max_step_units=args.max_step,
            joint_offset=args.offset,
            command_gripper=command_gripper,
            gripper_effort=args.gripper_effort,
            gripper_alpha=args.gripper_alpha,
            gripper_max_step_units=args.gripper_max_step,
            camera_buffers=camera_buffers,
            camera_tolerance_ms=args.camera_sync_tolerance_ms,
        )

    except KeyboardInterrupt:
        logging.info("Interrupted — saving current episode if any …")
        try:
            if dataset is not None and dataset.episode_buffer is not None:
                dataset.save_episode()
        except Exception:
            pass
    finally:
        _lock_and_disconnect(
            leader, follower,
            leader_connected=leader_connected,
            follower_connected=follower_connected,
            command_gripper=command_gripper,
            gripper_effort=args.gripper_effort,
        )
        for cam in cameras:
            cam.stop()
        logging.info("Disconnected. Dataset root: %s",
                     dataset.root if dataset is not None else "?")


if __name__ == "__main__":
    main()
