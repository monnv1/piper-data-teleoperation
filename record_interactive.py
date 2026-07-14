"""交互式遥操作录制 — ENTER 开始/结束，自动轮换 task JSON，归位 J5 最后动。

流程：
  1. 连接双机械臂 → 使能从臂 → 首次归位（主臂仅 MOVE J 锁定，不做重力补偿）
  2. 取第一条 task → 双机械臂归位（J5 最后动）→ 终端等候 ENTER
  3. 用户按主臂示教按钮 → ENTER 开始录制
  4. 用户完成任务 → 关示教 → ENTER 停止 → 保存渲染
  5. 取下一条 task → 归位 → 等候 ENTER（用户恢复场地）
  6. 最后一条完成后锁死关节，断开 CAN，退出

归位逻辑参考 deploy/tools/move_to_training_start.py：两阶段 MOVE J (J1-J4+J6 先，J5 后)。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import threading
import time
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
    HOME_SPEED,
    JOINT_OFFSET,
    LEADER_CAN,
    MAX_STEP_UNITS,
    make_piper,
    map_follower_target,
    read_gripper,
    read_joints,
)

DEFAULT_DEPLOY_CONFIG = Path("deploy/configs/piper_gemini_d435i.yaml")


# ── helpers (from record_lerobot.py) ────────────────────────────────────

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


def _read_gripper_m(piper) -> float:
    value = read_gripper(piper)
    return 0.0 if value is None else float(value) / 1_000_000.0


# ── dataset ─────────────────────────────────────────────────────────────

def _build_features(config, use_cameras: bool) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
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
    if root is not None and root.exists():
        if not args.force:
            raise FileExistsError(
                f"Dataset root already exists: {root}. Re-run with --force."
            )
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="piper",
        features=_build_features(config, use_cameras),
        use_videos=use_cameras,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


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


def _camera_frames_for_timestamp(buffers: dict[str, FrameBuffer],
                                 timestamp_ns: int,
                                 tolerance_ms: float) -> dict[str, np.ndarray]:
    tolerance_ns = int(tolerance_ms * 1_000_000.0)
    images = {}
    for name, buf in buffers.items():
        frame = buf.nearest(timestamp_ns, tolerance_ns)
        if frame is None:
            raise TimeoutError(
                f"No synced frame for {name} within {tolerance_ms:.1f} ms"
            )
        images[f"observation.images.{name}"] = np.asarray(frame.rgb).copy()
    return images


# ── arm lifecycle ───────────────────────────────────────────────────────
# Leader:  MOVE J position lock only — NO gravity compensation.
#          User enables teach mode via the physical button.
# Follower: MOVE J position lock, switches to 0xAD high-follow during recording.
# Homing: two-phase MOVE J, J5 last (ref: deploy/tools/move_to_training_start.py).

# ── arm status (ref: deploy/tools/move_to_training_start.py) ──────────

STATUS_NAMES = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_ik_solution",
    0x03: "singularity",
    0x04: "target_joint_limit",
    0x05: "joint_communication_error",
    0x06: "joint_brake_closed",
    0x07: "collision",
    0x08: "teach_overspeed",
    0x09: "joint_error",
    0x0A: "other_error",
}


def _read_arm_status(piper) -> dict[str, int]:
    wrapper = piper.GetArmStatus()
    status = getattr(wrapper, "arm_status", wrapper)
    return {
        "ctrl_mode": int(getattr(status, "ctrl_mode", -1)),
        "arm_status": int(getattr(status, "arm_status", -1)),
        "mode_feed": int(getattr(status, "mode_feed", -1)),
        "teach_status": int(getattr(status, "teach_status", -1)),
        "motion_status": int(getattr(status, "motion_status", -1)),
    }


def _arm_status_code(piper) -> int:
    return _read_arm_status(piper)["arm_status"]


def _enable_status(piper) -> list[bool]:
    try:
        return [bool(x) for x in piper.GetArmEnableStatus()]
    except Exception:
        return []


def _format_arm_state(piper) -> str:
    status = _read_arm_status(piper)
    enable = _enable_status(piper)
    enabled = "unknown" if not enable else "".join("1" if x else "0" for x in enable)
    return (
        f"ctrl=0x{status['ctrl_mode']:02X} "
        f"arm=0x{status['arm_status']:02X} "
        f"mode=0x{status['mode_feed']:02X} "
        f"teach=0x{status['teach_status']:02X} "
        f"enable={enabled}"
    )


def _check_status(piper, label: str) -> None:
    code = _arm_status_code(piper)
    if code != 0x00:
        name = STATUS_NAMES.get(code, "unknown")
        raise RuntimeError(f"{label}: arm_status=0x{code:02X} ({name})")


def _recover_from_target_joint_limit(piper) -> None:
    """Recover from 0x04 by holding current position + continuing trajectory.

    Ref: deploy/tools/move_to_training_start.py hold_current_position + resume_position_control."""
    code = _arm_status_code(piper)
    if code != 0x04:
        return
    logging.warning("Follower arm_status=0x04 (target_joint_limit) — recovering …")
    current_deg, _ = _read_joint_degrees_and_hz(piper)
    target_units = np.rint(current_deg * 1000.0).astype(int)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    piper.JointCtrl(*target_units.tolist())
    time.sleep(0.2)
    piper.MotionCtrl_1(0x00, 0x02, 0x00)  # continue trajectory execution
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    piper.JointCtrl(*target_units.tolist())
    time.sleep(0.2)
    code = _arm_status_code(piper)
    if code != 0x00:
        name = STATUS_NAMES.get(code, "unknown")
        raise RuntimeError(f"Recovery failed: arm_status=0x{code:02X} ({name})")
    logging.info("Recovered to normal status")


def _read_joint_degrees_and_hz(piper) -> tuple[np.ndarray, float]:
    message = piper.GetArmJointMsgs()
    joint = message.joint_state
    values = np.asarray(
        [joint.joint_1, joint.joint_2, joint.joint_3,
         joint.joint_4, joint.joint_5, joint.joint_6],
        dtype=np.float64,
    ) / 1000.0
    return values, float(message.Hz)


def _wait_for_joint_feedback(piper, label: str, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current_deg, hz = _read_joint_degrees_and_hz(piper)
        if hz > 0.0 and np.isfinite(current_deg).all():
            return
        time.sleep(0.05)
    raise TimeoutError(f"{label}: no valid joint feedback ({_format_arm_state(piper)})")


def _wait_until_teach_off(piper, label: str, timeout_s: float) -> None:
    """Wait for physical drag teaching to stop before taking CAN control.

    Observed Piper firmware reports teach_status=0x01 while gravity
    compensation is active and teach_status=0x02 after the operator exits
    teaching.  Treat only 0x01 as still teaching; 0x00/0x02 are safe to
    reacquire after several consecutive samples.
    """
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    consecutive_off = 0
    while time.monotonic() < deadline:
        status = _read_arm_status(piper)
        if status["teach_status"] != 0x01:
            consecutive_off += 1
            if consecutive_off >= 5:
                return
        else:
            consecutive_off = 0
        now = time.monotonic()
        if now - last_print >= 2.0:
            logging.info(
                "%s waiting for physical teach mode to stop (%s)",
                label, _format_arm_state(piper),
            )
            last_print = now
        time.sleep(0.1)
    raise TimeoutError(f"{label}: teach mode did not stop within {timeout_s:.1f}s")


def _enable_until_all_motors(piper, label: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        piper.EnablePiper()
        time.sleep(0.2)
        enable = _enable_status(piper)
        code = _arm_status_code(piper)
        if enable and all(enable) and code == 0x00:
            return
        if code == 0x04:
            _recover_from_target_joint_limit(piper)
    raise TimeoutError(f"{label}: motors did not enable into normal status ({_format_arm_state(piper)})")


def _resume_position_control_at_current(piper, label: str) -> np.ndarray:
    """Clear old trajectories and hold the measured pose in MOVE J control."""
    _wait_for_joint_feedback(piper, label)
    current_deg, _ = _read_joint_degrees_and_hz(piper)
    target_units = np.rint(current_deg * 1000.0).astype(int).tolist()

    piper.MotionCtrl_1(0x00, 0x03, 0x00)
    time.sleep(0.1)
    piper.MotionCtrl_1(0x00, 0x04, 0x00)
    time.sleep(0.1)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    piper.JointCtrl(*target_units)
    time.sleep(0.2)
    piper.MotionCtrl_1(0x00, 0x02, 0x00)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    piper.JointCtrl(*target_units)
    time.sleep(0.2)

    code = _arm_status_code(piper)
    if code != 0x00:
        name = STATUS_NAMES.get(code, "unknown")
        raise RuntimeError(f"{label}: cannot enter position hold: 0x{code:02X} ({name})")
    return current_deg


def _reacquire_leader_after_teach(leader, timeout_s: float) -> None:
    logging.info("Reacquiring leader after physical teach mode ...")
    _wait_until_teach_off(leader, "leader", timeout_s)
    leader.MotionCtrl_1(0x00, 0x00, 0x02)
    time.sleep(0.1)
    _enable_until_all_motors(leader, "leader")
    held = _resume_position_control_at_current(leader, "leader")
    logging.info(
        "Leader locked at current joints [deg]: %s (%s)",
        np.round(held, 2).tolist(), _format_arm_state(leader),
    )


def _exit_mit_mode(piper) -> None:
    """Transition from MIT mode (0xAD) back to position-speed MOVE J control.

    Uses trajectory-clear (track_ctrl=0x03) instead of emergency-stop to
    avoid motor power loss. Ref: piper_sdk MotionCtrl_1 Byte1 track_ctrl."""
    # Hold current position via existing MIT mode
    joints = read_joints(piper)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0xAD)
    piper.JointCtrl(*joints)
    time.sleep(0.2)

    logging.info("Exiting MIT mode → clear trajectory + switch to position-speed")
    piper.MotionCtrl_1(0x00, 0x03, 0x00)  # clear current trajectory (motors stay enabled)
    time.sleep(0.1)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)  # MOVE J + position-speed
    piper.JointCtrl(*joints)
    time.sleep(0.2)
    logging.info("MIT mode exit complete")


def _hold_position(piper) -> None:
    """Lock arm at current measured joint position (MOVE J hold).

    Recovers from common fault states (0x04 target_joint_limit,
    0x05 joint_communication_error) before commanding."""
    for attempt in range(3):
        code = _arm_status_code(piper)
        if code == 0x00:
            break
        if code == 0x04:
            _recover_from_target_joint_limit(piper)
        elif code in (0x05, 0x06):
            logging.warning("Arm status 0x%02X — retrying EnablePiper ...", code)
            piper.EnablePiper()
            time.sleep(0.5)
        else:
            name = STATUS_NAMES.get(code, "unknown")
            raise RuntimeError(f"Cannot hold position: arm_status=0x{code:02X} ({name})")
    joints = read_joints(piper)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    piper.JointCtrl(*joints)
    time.sleep(0.2)


def _read_ctrl_mode(piper) -> int:
    """Read ctrl_mode from arm status.  Returns -1 if unavailable."""
    wrapper = piper.GetArmStatus()
    status = getattr(wrapper, "arm_status", wrapper)
    return int(getattr(status, "ctrl_mode", -1))


def _init_follower(follower, command_gripper: bool, gripper_effort: int) -> None:
    """Always enable motors, then MOVE J position-lock.

    Startup ctrl_mode/arm_status can be stale SDK defaults — never trust them
    without an explicit EnablePiper first."""
    follower.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.2)

    logging.info("Enabling follower motors ...")
    _enable_until_all_motors(follower, "follower")

    _hold_position(follower)
    if command_gripper:
        g = read_gripper(follower)
        if g is not None:
            follower.GripperCtrl(g, gripper_effort, 0x02, 0)
            time.sleep(0.05)
            follower.GripperCtrl(g, gripper_effort, 0x03, 0)
            time.sleep(0.15)


def _init_leader(leader) -> None:
    """Always enable motors, then Standby.

    Startup ctrl_mode/arm_status can be stale SDK defaults — never trust them
    without an explicit EnablePiper first."""
    leader.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.2)

    logging.info("Enabling leader motors ...")
    _enable_until_all_motors(leader, "leader")
    _resume_position_control_at_current(leader, "leader startup")


def _move_to_target(piper, target_deg: np.ndarray, speed: int,
                    timeout_s: float = 30.0) -> None:
    """Closed-loop MOVE J to a joint target, blocking until settled.

    Assumes motors are already enabled and arm is in normal status (0x00)
    before calling — recovery happens upstream."""
    target_units = np.rint(target_deg * 1000.0).astype(int).tolist()
    _enable_until_all_motors(piper, "homing")
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    deadline = time.monotonic() + timeout_s
    settled = 0
    last_print = 0.0
    best_err = float("inf")
    best_err_time = time.monotonic()
    recovered_once = False
    while time.monotonic() < deadline:
        code = _arm_status_code(piper)
        if code != 0x00:
            name = STATUS_NAMES.get(code, "unknown")
            raise RuntimeError(f"Arm status 0x{code:02X} ({name}) during homing")

        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*target_units)
        current = np.asarray(read_joints(piper), dtype=np.float64) / 1000.0
        err = float(np.max(np.abs(current - target_deg)))
        if err <= 1.5:
            settled += 1
            if settled >= 5:
                return
        else:
            settled = 0

        now = time.monotonic()
        if err < best_err - 0.2:
            best_err = err
            best_err_time = now
        elif now - best_err_time > 6.0 and err > 1.5 and not recovered_once:
            logging.warning(
                "  homing not converging; reacquiring position control (%s)",
                _format_arm_state(piper),
            )
            _resume_position_control_at_current(piper, "homing recovery")
            piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
            best_err = float("inf")
            best_err_time = now
            recovered_once = True

        if now - last_print >= 2.0:
            logging.info("  homing: err=%.2f deg, status=0x%02X", err, code)
            last_print = now
        time.sleep(0.1)
    raise TimeoutError(f"Timed out moving to joints: {target_deg.tolist()}")


def _move_home_phased(piper, target_deg: np.ndarray, speed: int) -> None:
    """Two-phase MOVE J: J1-J4+J6 first, J5 last.

    Ref: deploy/tools/move_to_training_start.py — avoids striking the table."""
    _resume_position_control_at_current(piper, "pre-home")
    current = np.asarray(read_joints(piper), dtype=np.float64) / 1000.0

    # Phase 1: all except J5 (index 4)
    phase1 = target_deg.copy()
    phase1[4] = current[4]
    _move_to_target(piper, phase1, speed)

    # Phase 2: J5 last
    _move_to_target(piper, target_deg, speed)


def _move_both_to_home_phased(leader, follower, home_deg: list[float],
                              command_gripper: bool,
                              gripper_effort: int,
                              home_speed: int = HOME_SPEED) -> None:
    """Move both arms to home_deg (J5 last), then lock at home.

    Ref: deploy/tools/move_to_training_start.py."""
    home = np.asarray(home_deg, dtype=np.float64)
    home_units = [int(d * 1000) for d in home_deg]

    logging.info("Moving follower to home (J5 last) %s …", home_deg)
    _move_home_phased(follower, home, home_speed)
    follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    follower.JointCtrl(*home_units)
    if command_gripper:
        hg = read_gripper(follower)
        if hg is not None:
            follower.GripperCtrl(hg, gripper_effort, 0x01, 0)
    time.sleep(0.2)

    logging.info("Moving leader to home (J5 last) %s ...", home_deg)
    _move_home_phased(leader, home, home_speed)
    leader.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    leader.JointCtrl(*home_units)
    time.sleep(0.2)

    logging.info("Both arms at home and locked -- ready")


def _lock_and_disconnect(leader, follower, *, leader_connected: bool,
                         follower_connected: bool, command_gripper: bool,
                         gripper_effort: int) -> None:
    """Lock follower, leave leader in MOVE J hold, disconnect CAN."""
    try:
        if follower_connected:
            try:
                _hold_position(follower)
                if command_gripper:
                    hg = read_gripper(follower)
                    if hg is not None:
                        follower.GripperCtrl(hg, gripper_effort, 0x01, 0)
                time.sleep(0.2)
            except Exception as e:
                logging.warning("Could not lock follower: %s", e)
        if leader_connected:
            try:
                _hold_position(leader)
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


# ── interactive recording ──────────────────────────────────────────────

def _keyboard_listener(stop_event: threading.Event) -> None:
    try:
        input()
        stop_event.set()
    except EOFError:
        pass


def _record_one_episode(
    *,
    dataset,
    leader,
    follower,
    fk,
    frames: PiperFrameTransform,
    camera_buffers: dict[str, FrameBuffer],
    task: str,
    fps: int,
    speed_percent: int,
    alpha: float,
    max_step_units: int,
    joint_offset: list[int],
    command_gripper: bool,
    gripper_effort: int,
    gripper_alpha: float,
    gripper_max_step_units: int,
    camera_tolerance_ms: float,
) -> int:
    """Record one episode: ENTER to start, ENTER to stop.

    Mirrors record_lerobot._record_episode except the fixed duration is
    replaced by an interactive stop_event."""
    stop_event = threading.Event()

    print(f"\n  TASK: {task}")
    input("  Press ENTER to START recording …")
    logging.info("Recording — press ENTER to stop")

    listener = threading.Thread(target=_keyboard_listener, args=(stop_event,), daemon=True)
    listener.start()

    period = 1.0 / fps
    next_tick = time.monotonic()
    last_filtered: list[int] | None = None
    last_gripper_units: int | None = None
    last_gripper_log = 0.0
    n_frames = 0

    if command_gripper:
        g0 = read_gripper(leader)
        if g0 is not None:
            # Same sequence as teleop.py: clear while disabled, then enable and clear errors.
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
            logging.warning("Leader gripper feedback unavailable at recording start")

    follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
    while not stop_event.is_set():
        host_timestamp_ns = time.monotonic_ns()
        raw = read_joints(leader)
        follower_raw = read_joints(follower)
        leader_joints_deg = (np.asarray(raw, dtype=np.float32) / 1000.0).astype(np.float32)
        follower_joints_deg = (np.asarray(follower_raw, dtype=np.float32) / 1000.0).astype(np.float32)
        if last_filtered is None:
            last_filtered = raw[:]

        filtered = []
        for i in range(6):
            val = int(last_filtered[i] * (1.0 - alpha) + raw[i] * alpha)
            delta = val - last_filtered[i]
            delta = max(-max_step_units, min(max_step_units, delta))
            filtered.append(last_filtered[i] + delta)
        last_filtered = filtered
        target_joints = map_follower_target(filtered, joint_offset)

        state = _model_pose_from_sdk_units(_read_sdk_pose_units(follower), frames)
        action_pose = _model_pose_from_target_joints(fk, target_joints, frames)
        gripper_units = read_gripper(leader)
        if gripper_units is None:
            action_gripper_m = _read_gripper_m(follower)
        else:
            if last_gripper_units is None:
                last_gripper_units = gripper_units
            fg = int(last_gripper_units * (1.0 - gripper_alpha)
                     + gripper_units * gripper_alpha)
            dg = fg - last_gripper_units
            dg = max(-gripper_max_step_units, min(gripper_max_step_units, dg))
            last_gripper_units = last_gripper_units + dg
            action_gripper_m = float(last_gripper_units) / 1_000_000.0

        frame = {
            "observation.state": state,
            "observation.leader_joints": leader_joints_deg,
            "observation.follower_joints": follower_joints_deg,
            "action": np.concatenate(
                [action_pose, np.asarray([action_gripper_m], dtype=np.float32)]
            ).astype(np.float32),
        }
        frame.update(
            _camera_frames_for_timestamp(camera_buffers, host_timestamp_ns, camera_tolerance_ms)
        )
        dataset.add_frame(frame, task=task)
        n_frames += 1

        follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
        follower.JointCtrl(*target_joints)
        if command_gripper and last_gripper_units is not None:
            follower.GripperCtrl(last_gripper_units, gripper_effort, 0x01, 0)

        now = time.monotonic()
        if command_gripper and now - last_gripper_log >= 2.0:
            follower_gripper = read_gripper(follower)
            leader_text = "None" if gripper_units is None else f"{gripper_units / 1000.0:.1f}mm"
            target_text = "None" if last_gripper_units is None else f"{last_gripper_units / 1000.0:.1f}mm"
            follower_text = "None" if follower_gripper is None else f"{follower_gripper / 1000.0:.1f}mm"
            logging.info(
                "gripper leader=%s target=%s follower=%s effort=%d",
                leader_text, target_text, follower_text, gripper_effort,
            )
            last_gripper_log = now

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    _exit_mit_mode(follower)
    # Arms are already stationary (user has dragged leader to lying-down pose
    # and turned off teach mode).  No leader mode change here -- just save.
    logging.info("Stopped — saving episode (%d frames) …", n_frames)
    dataset.save_episode()
    logging.info("Episode saved and video rendered.")
    return n_frames


# ── main ────────────────────────────────────────────────────────────────

def _load_tasks(tasks_file: str) -> list[str]:
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    raise ValueError("Tasks JSON must be a list or {\"tasks\": [...]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive teleop recording — ENTER to start/stop each episode"
    )
    parser.add_argument("--repo-id", default="local/piper-teleop")
    parser.add_argument("--root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY_CONFIG))
    parser.add_argument("--leader-can", default=LEADER_CAN)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-step", type=int, default=MAX_STEP_UNITS)
    parser.add_argument("--offset", type=int, nargs=6, default=JOINT_OFFSET)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--camera-sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--camera-startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--gripper-alpha", type=float, default=GRIPPER_ALPHA)
    parser.add_argument("--gripper-max-step", type=int, default=GRIPPER_MAX_STEP_UNITS)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--video-codec", default="h264",
                        choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--no-home-between", action="store_true",
                        help="Skip home return between episodes")
    parser.add_argument("--home-deg", type=float, nargs=6,
                        default=[0.0, 70.913, -60.913, 0.0, 58.398, 0.0],
                        help="Home joint angles in degrees (default: J5-last safe position)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat task list N times")
    parser.add_argument("--teach-off-timeout-s", type=float, default=30.0,
                        help="Seconds to wait for physical teach mode to turn off after each episode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    if args.tasks_file:
        all_tasks = _load_tasks(args.tasks_file)
        logging.info("Loaded %d tasks from %s", len(all_tasks), args.tasks_file)
    else:
        all_tasks = [args.task]

    tasks = all_tasks * args.repeat
    total = len(tasks)
    logging.info("Will record %d episodes total", total)

    config = load_config(args.deploy_config)
    use_cameras = not args.no_cameras
    dataset = _create_dataset(args, config, use_cameras)

    from piper_sdk import C_PiperForwardKinematics
    frames = PiperFrameTransform(
        config.robot.sdk_to_model_translation_m,
        config.robot.sdk_to_model_euler_xyz_rad,
    )
    fk = C_PiperForwardKinematics(config.robot.dh_is_offset)

    camera_buffers, cameras = _start_cameras(config, use_cameras)
    leader = make_piper(args.leader_can)
    follower = make_piper(args.follower_can)
    leader_connected = False
    follower_connected = False
    command_gripper = not args.no_gripper

    try:
        if use_cameras:
            logging.info("Waiting for camera frames …")
            _wait_for_cameras(camera_buffers, args.camera_startup_timeout_s)

        # ── startup (once) ──
        logging.info("Connecting leader on %s", args.leader_can)
        leader.ConnectPort(piper_init=False, start_thread=True)
        leader_connected = True
        time.sleep(0.5)

        logging.info("Connecting follower on %s", args.follower_can)
        follower.ConnectPort(piper_init=False, start_thread=True)
        follower_connected = True
        time.sleep(0.5)

        # Leader: MOVE J hold only (user controls teach via physical button)
        _init_leader(leader)
        # Follower: MOVE J hold
        _init_follower(follower, command_gripper, args.gripper_effort)

        # ── episode loop ──
        for ep_idx, task in enumerate(tasks):
            print(f"\n{'─'*60}")
            print(f"  Episode {ep_idx + 1}/{total}")
            print(f"{'─'*60}")

            # Step 1: home both arms (J5 last, ref: move_to_training_start.py)
            if not args.no_home_between:
                _move_both_to_home_phased(leader, follower, args.home_deg,
                                          command_gripper,
                                          args.gripper_effort)

            # Step 2-4: wait → record → save
            n_frames = _record_one_episode(
                dataset=dataset,
                leader=leader,
                follower=follower,
                fk=fk,
                frames=frames,
                camera_buffers=camera_buffers,
                task=task,
                fps=args.fps,
                speed_percent=args.speed_percent,
                alpha=args.alpha,
                max_step_units=args.max_step,
                joint_offset=args.offset,
                command_gripper=command_gripper,
                gripper_effort=args.gripper_effort,
                gripper_alpha=args.gripper_alpha,
                gripper_max_step_units=args.gripper_max_step,
                camera_tolerance_ms=args.camera_sync_tolerance_ms,
            )
            # Follower stayed in MOVE J — just re-hold.  Leader may be stuck
            # after physical-teach exit — force re-enable like startup.
            if not args.no_home_between:
                _hold_position(follower)

                _reacquire_leader_after_teach(leader, args.teach_off_timeout_s)

            logging.info("Episode %d/%d done — %d frames, task: %s",
                         ep_idx + 1, total, n_frames, task)

        logging.info("All %d episodes complete — locking and disconnecting.", total)

    except KeyboardInterrupt:
        logging.info("Interrupted — saving current episode if it has unsaved frames ...")
        try:
            dataset.save_episode()
            logging.info("Current episode saved.")
        except ValueError as e:
            logging.info("No unsaved episode frames to save: %s", e)
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
        logging.info("Dataset root: %s", dataset.root)


if __name__ == "__main__":
    main()
