"""Record Piper teleoperation episodes directly as a LeRobot dataset.

This is intentionally independent from the official LeRobot recorder CLI. It
uses LeRobotDataset.create/add_frame/save_episode while reusing the local Piper
teleoperation loop and camera adapters.
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
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
    HOME_DEG,
    HOME_SPEED,
    GRIPPER_ALPHA,
    GRIPPER_MAX_STEP_UNITS,
    JOINT_OFFSET,
    LEADER_CAN,
    MAX_STEP_UNITS,
    make_piper,
    map_follower_target,
    read_gripper,
    read_joints,
    _move_to_home,
)


DEFAULT_DEPLOY_CONFIG = Path("deploy/configs/piper_gemini_d435i.yaml")


def _read_sdk_pose_units(piper) -> np.ndarray:
    pose = piper.GetArmEndPoseMsgs().end_pose
    return np.asarray(
        [
            pose.X_axis,
            pose.Y_axis,
            pose.Z_axis,
            pose.RX_axis,
            pose.RY_axis,
            pose.RZ_axis,
        ],
        dtype=np.float64,
    )


def _model_pose_from_sdk_units(
    pose_units: np.ndarray,
    frames: PiperFrameTransform,
) -> np.ndarray:
    sdk_position_m = pose_units[:3] / 1_000_000.0
    sdk_euler_rad = pose_units[3:] * (math.pi / 180_000.0)
    position_m, euler_rad = frames.sdk_to_model_pose(sdk_position_m, sdk_euler_rad)
    return np.concatenate(
        [position_m.astype(np.float32), euler_rad.astype(np.float32)]
    )


def _model_pose_from_target_joints(
    fk,
    target_joint_units: list[int],
    frames: PiperFrameTransform,
) -> np.ndarray:
    joint_radians = np.radians(np.asarray(target_joint_units, dtype=np.float64) / 1000.0)
    # Piper SDK FK returns link poses in mm and degrees.
    sdk_pose = np.asarray(fk.CalFK(joint_radians)[-1], dtype=np.float64)
    sdk_position_m = sdk_pose[:3] / 1000.0
    sdk_euler_rad = np.radians(sdk_pose[3:])
    position_m, euler_rad = frames.sdk_to_model_pose(sdk_position_m, sdk_euler_rad)
    return np.concatenate(
        [position_m.astype(np.float32), euler_rad.astype(np.float32)]
    )


def _read_gripper_m(piper) -> float:
    value = read_gripper(piper)
    return 0.0 if value is None else float(value) / 1_000_000.0


def _build_features(config, use_cameras: bool) -> dict:
    features = {
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
    if use_cameras:
        for name, camera in config.cameras.items():
            if not camera.enabled:
                continue
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channel"],
            }
    return features


def _configure_lerobot_video_codec(video_codec: str) -> None:
    if video_codec == "libsvtav1":
        return
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    import lerobot.datasets.video_utils as video_utils

    original_encode = video_utils.encode_video_frames

    def encode_video_frames_with_codec(imgs_dir, video_path, fps, **kwargs):
        kwargs.setdefault("vcodec", video_codec)
        return original_encode(imgs_dir, video_path, fps, **kwargs)

    video_utils.encode_video_frames = encode_video_frames_with_codec
    lerobot_dataset_module.encode_video_frames = encode_video_frames_with_codec


def _create_dataset(args, config, use_cameras: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _configure_lerobot_video_codec(args.video_codec)

    root = Path(args.root).expanduser() if args.root else None
    if root is not None and root.exists():
        if not args.force:
            raise FileExistsError(
                f"Dataset root already exists: {root}. Re-run with --force to overwrite."
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


def _start_cameras(config, enabled: bool) -> tuple[dict[str, FrameBuffer], list]:
    if not enabled:
        return {}, []
    buffers: dict[str, FrameBuffer] = {}
    cameras = []
    for name, camera_config in config.cameras.items():
        if not camera_config.enabled:
            continue
        buffer = FrameBuffer(maxlen=max(120, int(camera_config.fps * 5)))
        camera = create_camera(name, camera_config, buffer)
        camera.start()
        buffers[name] = buffer
        cameras.append(camera)
    return buffers, cameras


def _wait_for_cameras(buffers: dict[str, FrameBuffer], timeout_s: float) -> None:
    if not buffers:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(buffer.latest() is not None for buffer in buffers.values()):
            return
        time.sleep(0.05)
    missing = [name for name, buffer in buffers.items() if buffer.latest() is None]
    raise TimeoutError(f"Timed out waiting for camera frames: {missing}")


def _camera_frames_for_timestamp(
    buffers: dict[str, FrameBuffer],
    timestamp_ns: int,
    tolerance_ms: float,
) -> dict[str, np.ndarray]:
    tolerance_ns = int(tolerance_ms * 1_000_000.0)
    images = {}
    for name, buffer in buffers.items():
        frame = buffer.nearest(timestamp_ns, tolerance_ns)
        if frame is None:
            raise TimeoutError(
                f"No synchronized frame for {name} within {tolerance_ms:.1f} ms"
            )
        images[f"observation.images.{name}"] = np.asarray(frame.rgb).copy()
    return images


def _enable_teleop(leader, follower, command_gripper: bool, gripper_effort: int) -> list[int]:
    logging.info("Configuring both arms as slave mode")
    leader.MasterSlaveConfig(0xFC, 0, 0, 0)
    follower.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.2)

    logging.info("Enabling leader for gravity compensation")
    if not leader.EnablePiper():
        time.sleep(0.5)
        leader.EnablePiper()
    # Standby: motors enabled + gravity compensation, no position lock.
    leader.MotionCtrl_2(0x00, 0x01, 1, 0x00)
    time.sleep(0.3)

    home = read_joints(follower)
    logging.info("Enabling follower and holding current position")
    if not follower.EnablePiper():
        time.sleep(0.5)
        follower.EnablePiper()
    time.sleep(0.2)
    follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    follower.JointCtrl(*home)
    time.sleep(0.2)

    if command_gripper:
        gripper = read_gripper(leader)
        if gripper is not None:
            follower.GripperCtrl(gripper, gripper_effort, 0x02, 0)
            time.sleep(0.05)
            follower.GripperCtrl(gripper, gripper_effort, 0x03, 0)
            time.sleep(0.15)
        else:
            logging.warning("Leader gripper feedback unavailable at startup")
    return home


def _restore_and_disconnect(
    leader,
    follower,
    *,
    leader_connected: bool,
    follower_connected: bool,
    command_gripper: bool,
    gripper_effort: int,
    do_home: bool,
) -> None:
    try:
        if follower_connected:
            try:
                if do_home:
                    home_units = [int(deg * 1000) for deg in HOME_DEG]
                    logging.info("Moving follower to home %s", HOME_DEG)
                    _move_to_home(follower, home_units, HOME_SPEED)
                else:
                    hold = read_joints(follower)
                    follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
                    follower.JointCtrl(*hold)
                    time.sleep(0.2)
                if command_gripper:
                    hold_gripper = read_gripper(follower)
                    if hold_gripper is not None:
                        follower.GripperCtrl(hold_gripper, gripper_effort, 0x01, 0)
                        time.sleep(0.1)
            except Exception as error:
                logging.warning("Could not restore follower before disconnect: %s", error)

        if leader_connected:
            try:
                if do_home:
                    home_units = [int(deg * 1000) for deg in HOME_DEG]
                    logging.info("Moving leader to home %s", HOME_DEG)
                    leader.MotionCtrl_2(0x01, 0x01, HOME_SPEED, 0x00)
                    _move_to_home(leader, home_units, HOME_SPEED)
                else:
                    # Leave the leader in the same standby/gravity-compensation mode used by teleop.
                    leader.MotionCtrl_2(0x00, 0x01, 1, 0x00)
                    time.sleep(0.1)
            except Exception as error:
                logging.warning("Could not restore leader before disconnect: %s", error)
    finally:
        if follower_connected:
            try:
                follower.DisconnectPort()
            except Exception as error:
                logging.warning("Could not disconnect follower: %s", error)
        if leader_connected:
            try:
                leader.DisconnectPort()
            except Exception as error:
                logging.warning("Could not disconnect leader: %s", error)


def _record_episode(
    *,
    dataset,
    leader,
    follower,
    fk,
    frames: PiperFrameTransform,
    camera_buffers: dict[str, FrameBuffer],
    task: str,
    duration_s: float,
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
    period = 1.0 / fps
    deadline = time.monotonic() + duration_s
    next_tick = time.monotonic()
    last_filtered: list[int] | None = None
    last_gripper_units: int | None = None
    n_frames = 0

    follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
    while time.monotonic() < deadline:
        host_timestamp_ns = time.monotonic_ns()
        raw = read_joints(leader)
        if last_filtered is None:
            last_filtered = raw[:]

        filtered = []
        for index in range(6):
            value = int(last_filtered[index] * (1.0 - alpha) + raw[index] * alpha)
            delta = value - last_filtered[index]
            delta = max(-max_step_units, min(max_step_units, delta))
            filtered.append(last_filtered[index] + delta)
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
            filtered_gripper = int(
                last_gripper_units * (1.0 - gripper_alpha)
                + gripper_units * gripper_alpha
            )
            delta_gripper = filtered_gripper - last_gripper_units
            delta_gripper = max(
                -gripper_max_step_units,
                min(gripper_max_step_units, delta_gripper),
            )
            last_gripper_units = last_gripper_units + delta_gripper
            action_gripper_m = float(last_gripper_units) / 1_000_000.0

        frame = {
            "observation.state": state,
            "action": np.concatenate(
                [action_pose, np.asarray([action_gripper_m], dtype=np.float32)]
            ).astype(np.float32),
        }
        frame.update(
            _camera_frames_for_timestamp(
                camera_buffers,
                host_timestamp_ns,
                camera_tolerance_ms,
            )
        )
        dataset.add_frame(frame, task=task)
        n_frames += 1

        follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
        follower.JointCtrl(*target_joints)
        if command_gripper and last_gripper_units is not None:
            follower.GripperCtrl(last_gripper_units, gripper_effort, 0x01, 0)

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    dataset.save_episode()
    return n_frames





def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record Piper teleoperation episodes in LeRobot format"
    )
    parser.add_argument("--repo-id", default="local/piper-teleop")
    parser.add_argument("--root", default=None, help="Dataset root. Defaults to HF_LEROBOT_HOME/repo-id")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing dataset root")
    parser.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY_CONFIG))
    parser.add_argument("--leader-can", default=LEADER_CAN)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--start-immediately", action="store_true",
                        help="Do not wait for ENTER before each episode")
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-step", type=int, default=MAX_STEP_UNITS)
    parser.add_argument("--offset", type=int, nargs=6, default=JOINT_OFFSET)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--camera-sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--camera-startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--home", action="store_true",
                        help="Move both arms to teleop HOME_DEG before disconnecting")
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--gripper-alpha", type=float, default=GRIPPER_ALPHA)
    parser.add_argument("--gripper-max-step", type=int, default=GRIPPER_MAX_STEP_UNITS)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1"],
                        help="Codec used when LeRobot encodes camera frames to mp4")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be in [1, 100]")
    if not args.start_immediately and not sys.stdin.isatty():
        raise RuntimeError(
            "This recorder waits for ENTER before each episode, but stdin is not "
            "interactive. Run it as `python3 -m teleoperation.record_lerobot` "
            "inside the activated conda environment, or add --start-immediately."
        )

    config = load_config(args.deploy_config)
    use_cameras = not args.no_cameras
    dataset = _create_dataset(args, config, use_cameras)

    from piper_sdk import C_PiperForwardKinematics  # type: ignore

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
    try:
        if use_cameras:
            logging.info("Waiting for camera frames")
            _wait_for_cameras(camera_buffers, args.camera_startup_timeout_s)

        logging.info("Connecting leader on %s", args.leader_can)
        leader.ConnectPort()
        leader_connected = True
        time.sleep(0.5)
        logging.info("Connecting follower on %s", args.follower_can)
        follower.ConnectPort()
        follower_connected = True
        time.sleep(0.5)
        _enable_teleop(leader, follower, not args.no_gripper, args.gripper_effort)

        for episode_index in range(args.episodes):
            if args.start_immediately:
                logging.info("Starting episode %d/%d", episode_index + 1, args.episodes)
            else:
                input(f"Press ENTER to start episode {episode_index + 1}/{args.episodes}...")
            logging.info("Recording episode %d for %.1f s", episode_index, args.duration_s)
            n_frames = _record_episode(
                dataset=dataset,
                leader=leader,
                follower=follower,
                fk=fk,
                frames=frames,
                camera_buffers=camera_buffers,
                task=args.task,
                duration_s=args.duration_s,
                fps=args.fps,
                speed_percent=args.speed_percent,
                alpha=args.alpha,
                max_step_units=args.max_step,
                joint_offset=args.offset,
                command_gripper=not args.no_gripper,
                gripper_effort=args.gripper_effort,
                gripper_alpha=args.gripper_alpha,
                gripper_max_step_units=args.gripper_max_step,
                camera_tolerance_ms=args.camera_sync_tolerance_ms,
            )
            logging.info("Saved episode %d with %d frames", episode_index, n_frames)
    except KeyboardInterrupt:
        logging.info("Stopped by user — saving current episode...")
        dataset.save_episode()
        logging.info("Current episode saved.")
    finally:
        _restore_and_disconnect(
            leader,
            follower,
            leader_connected=leader_connected,
            follower_connected=follower_connected,
            command_gripper=not args.no_gripper,
            gripper_effort=args.gripper_effort,
            do_home=args.home,
        )
        for camera in cameras:
            camera.stop()
        logging.info("Dataset root: %s", dataset.root)


if __name__ == "__main__":
    main()
