import json
from types import SimpleNamespace

import numpy as np
import pytest

import teleoperation.record_continuous as recorder
from deploy.common.latest import FrameBuffer
from deploy.common.messages import CameraFrame
from lerobot.datasets.utils import validate_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from teleoperation.record_continuous import (
    _append_episode_timing_metadata,
    _build_features,
    _camera_observation,
    _summarize_episode_timing,
)


def _config():
    return SimpleNamespace(
        cameras={
            "wrist_cam": SimpleNamespace(
                enabled=True,
                height=480,
                width=640,
                fps=30,
            )
        }
    )


def test_features_keep_lerobot_keys_and_add_real_timing():
    features = _build_features(_config(), use_cameras=True)

    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["shape"] == (7,)
    assert features["recording.sample_monotonic_ns"]["dtype"] == "int64"
    assert features["recording.actual_fps"]["dtype"] == "float32"
    assert "recording.camera_host_timestamp_ns.wrist_cam" in features
    assert "recording.camera_frame_number.wrist_cam" in features


def test_non_camera_frame_validates_with_lerobot_schema():
    features = _build_features(_config(), use_cameras=False)
    frame = {
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.leader_joints": np.zeros(6, dtype=np.float32),
        "observation.follower_joints": np.zeros(6, dtype=np.float32),
        "action": np.zeros(7, dtype=np.float32),
        "recording.sample_monotonic_ns": np.zeros(1, dtype=np.int64),
        "recording.sample_wall_time_ns": np.zeros(1, dtype=np.int64),
        "recording.observation_monotonic_ns": np.zeros(1, dtype=np.int64),
        "recording.action_sent_monotonic_ns": np.zeros(1, dtype=np.int64),
        "recording.actual_dt_s": np.zeros(1, dtype=np.float32),
        "recording.actual_fps": np.zeros(1, dtype=np.float32),
        "recording.target_fps": np.zeros(1, dtype=np.float32),
    }

    validate_frame(frame, features)


def test_real_timing_fields_save_in_official_lerobot_dataset(tmp_path):
    features = _build_features(_config(), use_cameras=False)
    dataset = LeRobotDataset.create(
        repo_id="local/timing-test",
        root=tmp_path / "dataset",
        fps=40,
        robot_type="piper",
        features=features,
        use_videos=False,
    )
    for index in range(2):
        timestamp_ns = 1_000_000_000 + index * 25_000_000
        frame = {
            "observation.state": np.zeros(7, dtype=np.float32),
            "observation.leader_joints": np.zeros(6, dtype=np.float32),
            "observation.follower_joints": np.zeros(6, dtype=np.float32),
            "action": np.zeros(7, dtype=np.float32),
            "recording.sample_monotonic_ns": np.asarray(
                [timestamp_ns], dtype=np.int64
            ),
            "recording.sample_wall_time_ns": np.asarray(
                [timestamp_ns], dtype=np.int64
            ),
            "recording.observation_monotonic_ns": np.asarray(
                [timestamp_ns + 1], dtype=np.int64
            ),
            "recording.action_sent_monotonic_ns": np.asarray(
                [timestamp_ns + 2], dtype=np.int64
            ),
            "recording.actual_dt_s": np.asarray(
                [0.0 if index == 0 else 0.025], dtype=np.float32
            ),
            "recording.actual_fps": np.asarray(
                [0.0 if index == 0 else 40.0], dtype=np.float32
            ),
            "recording.target_fps": np.asarray([40.0], dtype=np.float32),
        }
        dataset.add_frame(frame, task="test")

    dataset.save_episode()

    assert list((tmp_path / "dataset" / "data").rglob("*.parquet"))


def test_camera_observation_is_complete_and_timestamped():
    buffer = FrameBuffer()
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    buffer.append(
        CameraFrame(
            camera="wrist_cam",
            serial="camera",
            frame_number=42,
            device_timestamp_ms=123.5,
            host_timestamp_ns=1_000_000_000,
            rgb=rgb,
        )
    )

    observation = _camera_observation(
        {"wrist_cam": buffer},
        observation_timestamp_ns=1_010_000_000,
        tolerance_ms=50.0,
    )

    assert observation is not None
    np.testing.assert_array_equal(observation["observation.images.wrist_cam"], rgb)
    assert observation["recording.camera_frame_number.wrist_cam"].item() == 42
    assert observation["recording.camera_age_ms.wrist_cam"].item() == pytest.approx(10.0)


def test_stale_camera_rejects_whole_sample():
    buffer = FrameBuffer()
    buffer.append(
        CameraFrame(
            camera="wrist_cam",
            serial="camera",
            frame_number=1,
            device_timestamp_ms=0.0,
            host_timestamp_ns=1_000_000_000,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )

    observation = _camera_observation(
        {"wrist_cam": buffer},
        observation_timestamp_ns=1_100_000_000,
        tolerance_ms=50.0,
    )

    assert observation is None


def test_episode_timing_reports_measured_rate():
    summary = _summarize_episode_timing(
        [1_000_000_000, 1_025_000_000, 1_050_000_000],
        target_fps=40,
        skipped_camera_samples=2,
        skipped_gripper_samples=3,
        control_overruns=1,
    )

    assert summary.frames == 3
    assert summary.measured_fps == pytest.approx(40.0)
    assert summary.median_dt_s == pytest.approx(0.025)
    assert summary.skipped_camera_samples == 2
    assert summary.skipped_gripper_samples == 3
    assert summary.control_overruns == 1


def test_episode_timing_sidecar_documents_timestamp_semantics(tmp_path):
    summary = _summarize_episode_timing(
        [1_000_000_000, 1_025_000_000],
        target_fps=40,
        skipped_camera_samples=0,
        skipped_gripper_samples=0,
        control_overruns=0,
    )

    _append_episode_timing_metadata(tmp_path, 0, "pick cube", summary)

    path = tmp_path / "meta" / "recording_timing.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["measured_fps"] == pytest.approx(40.0)
    assert "recording.sample_monotonic_ns" in payload["timestamp_semantics"]


def test_control_loop_uses_lerobot_observation_action_send_order(monkeypatch):
    calls = []

    class Arm:
        def __init__(self, name):
            self.name = name

        def MotionCtrl_2(self, *_args):
            calls.append(f"{self.name}.motion")

        def JointCtrl(self, *_args):
            calls.append(f"{self.name}.send")
            state.stop_event.set()

    leader = Arm("leader")
    follower = Arm("follower")
    state = recorder._FollowState()

    def fake_read_joints(arm):
        calls.append(f"{arm.name}.joints")
        return [0, 0, 0, 0, 0, 0]

    monkeypatch.setattr(recorder, "read_joints", fake_read_joints)
    monkeypatch.setattr(recorder, "read_gripper", lambda _arm: None)
    monkeypatch.setattr(
        recorder,
        "_read_sdk_pose_units",
        lambda _arm: calls.append("follower.pose") or np.zeros(6),
    )
    monkeypatch.setattr(
        recorder,
        "_model_pose_from_sdk_units",
        lambda _pose, _frames: np.zeros(6, dtype=np.float32),
    )
    monkeypatch.setattr(
        recorder,
        "_read_gripper_m",
        lambda _arm: calls.append("follower.gripper") or 0.0,
    )

    recorder._follow_loop(
        leader=leader,
        follower=follower,
        state=state,
        dataset=None,
        fps=30,
        speed_percent=10,
        alpha=1.0,
        max_step_units=2_000,
        joint_offset=[0, 0, 0, 0, 0, 0],
        command_gripper=False,
        gripper_effort=1_000,
        gripper_alpha=1.0,
        gripper_max_step_units=1_000,
        camera_buffers={},
        camera_tolerance_ms=50.0,
        fk=None,
        frames=None,
    )

    assert calls == [
        "follower.joints",
        "follower.pose",
        "follower.gripper",
        "leader.joints",
        "follower.motion",
        "follower.send",
    ]


def test_control_loop_records_measured_follower_and_sent_gripper_separately(
    monkeypatch,
):
    class Arm:
        def __init__(self, name):
            self.name = name

        def MotionCtrl_2(self, *_args):
            pass

        def JointCtrl(self, *_args):
            state.stop_event.set()

        def GripperCtrl(self, *_args):
            pass

    class Dataset:
        def __init__(self):
            self.frames = []

        def add_frame(self, frame, task):
            self.frames.append((frame, task))

    leader = Arm("leader")
    follower = Arm("follower")
    dataset = Dataset()
    state = recorder._FollowState()
    state.begin_episode("test")

    monkeypatch.setattr(recorder, "read_joints", lambda _arm: [0] * 6)
    monkeypatch.setattr(
        recorder,
        "read_gripper",
        lambda arm: 50_000 if arm is leader else 10_000,
    )
    monkeypatch.setattr(recorder, "_read_sdk_pose_units", lambda _arm: np.zeros(6))
    monkeypatch.setattr(
        recorder,
        "_model_pose_from_sdk_units",
        lambda _pose, _frames: np.zeros(6, dtype=np.float32),
    )
    monkeypatch.setattr(
        recorder,
        "_model_pose_from_target_joints",
        lambda _fk, _target, _frames: np.zeros(6, dtype=np.float32),
    )

    recorder._follow_loop(
        leader=leader,
        follower=follower,
        state=state,
        dataset=dataset,
        fps=40,
        speed_percent=10,
        alpha=1.0,
        max_step_units=2_000,
        joint_offset=[0] * 6,
        command_gripper=True,
        gripper_effort=1_000,
        gripper_alpha=1.0,
        gripper_max_step_units=100_000,
        camera_buffers={},
        camera_tolerance_ms=50.0,
        fk=None,
        frames=None,
    )

    assert len(dataset.frames) == 1
    frame, task = dataset.frames[0]
    assert task == "test"
    assert frame["observation.state"][6] == pytest.approx(0.01)
    assert frame["action"][6] == pytest.approx(0.05)
    assert frame["observation.state"][6] != frame["action"][6]


def test_control_loop_skips_sample_without_follower_gripper_feedback(monkeypatch):
    class Arm:
        def __init__(self, name):
            self.name = name

        def MotionCtrl_2(self, *_args):
            pass

        def JointCtrl(self, *_args):
            state.stop_event.set()

    class Dataset:
        def __init__(self):
            self.frames = []

        def add_frame(self, frame, task):
            self.frames.append((frame, task))

    leader = Arm("leader")
    follower = Arm("follower")
    dataset = Dataset()
    state = recorder._FollowState()
    state.begin_episode("test")

    monkeypatch.setattr(recorder, "read_joints", lambda _arm: [0] * 6)
    monkeypatch.setattr(
        recorder,
        "read_gripper",
        lambda arm: 50_000 if arm is leader else None,
    )
    monkeypatch.setattr(recorder, "_read_sdk_pose_units", lambda _arm: np.zeros(6))
    monkeypatch.setattr(
        recorder,
        "_model_pose_from_sdk_units",
        lambda _pose, _frames: np.zeros(6, dtype=np.float32),
    )

    recorder._follow_loop(
        leader=leader,
        follower=follower,
        state=state,
        dataset=dataset,
        fps=40,
        speed_percent=10,
        alpha=1.0,
        max_step_units=2_000,
        joint_offset=[0] * 6,
        command_gripper=False,
        gripper_effort=1_000,
        gripper_alpha=1.0,
        gripper_max_step_units=100_000,
        camera_buffers={},
        camera_tolerance_ms=50.0,
        fk=None,
        frames=None,
    )

    assert dataset.frames == []
    assert state.skipped_gripper_samples == 1
