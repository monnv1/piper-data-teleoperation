"""Record Piper J6 joint and raw motor feedback for wraparound analysis.

This script only reads feedback and writes CSV. It does not enable motors,
change modes, or send any motion/gripper command. Put the arm in teaching mode
manually, start this script, rotate J6 left then right then back to center, and
press Ctrl+C to stop recording.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from teleoperation.teleop import LEADER_CAN, make_piper, read_joints


def deg(units: int | float) -> float:
    return float(units) / 1000.0


def rad(units: int | float) -> float:
    return float(units) / 1000.0


def rad_to_deg(value_rad: float) -> float:
    return value_rad * 180.0 / 3.141592653589793


def shortest_delta_units(current: int, previous: int) -> int:
    return ((current - previous + 180000) % 360000) - 180000


class JointJ6Unwrapper:
    def __init__(self, origin_units: int):
        self.origin_units = origin_units
        self.previous_units = origin_units
        self.unwrapped_delta_units = 0

    def update(self, current_units: int) -> tuple[int, int, int]:
        origin_delta = shortest_delta_units(current_units, self.origin_units)
        step_delta = shortest_delta_units(current_units, self.previous_units)
        self.unwrapped_delta_units += step_delta
        self.previous_units = current_units
        return origin_delta, step_delta, self.unwrapped_delta_units


class LinearUnwrapper:
    def __init__(self, origin_units: int):
        self.origin_units = origin_units
        self.previous_units = origin_units
        self.delta_units = 0

    def update(self, current_units: int) -> tuple[int, int, int]:
        origin_delta = current_units - self.origin_units
        step_delta = current_units - self.previous_units
        self.delta_units += step_delta
        self.previous_units = current_units
        return origin_delta, step_delta, self.delta_units


def get_attr(obj: Any, name: str, default: Any = "") -> Any:
    return getattr(obj, name, default)


def read_motor6(arm) -> Any:
    return arm.GetArmHighSpdInfoMsgs().motor_6


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one Piper arm's J6 joint and motor feedback to CSV")
    parser.add_argument("--can", default=LEADER_CAN, help="CAN interface to record, default: can1")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--note", default="", help="Optional note stored in every row")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.hz <= 0:
        raise ValueError("--hz must be positive")

    output = Path(args.output) if args.output else Path("recordings") / f"j6_record_{args.can}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Connecting %s for feedback only", args.can)
    arm = make_piper(args.can)
    arm.ConnectPort()

    fieldnames = [
        "sample",
        "time_s",
        "note",
        "j6_units",
        "j6_deg",
        "j6_delta_from_start_units",
        "j6_delta_from_start_deg",
        "j6_step_units",
        "j6_step_deg",
        "j6_unwrapped_delta_units",
        "j6_unwrapped_delta_deg",
        "motor6_pos_units",
        "motor6_pos_rad",
        "motor6_delta_from_start_units",
        "motor6_delta_from_start_rad",
        "motor6_step_units",
        "motor6_step_rad",
        "motor6_unwrapped_delta_units",
        "motor6_unwrapped_delta_rad",
        "motor6_speed_units",
        "motor6_speed_rad_s",
        "motor6_current_raw",
        "motor6_effort_raw",
        "motor6_can_id",
        "high_spd_hz",
    ]
    for i in range(1, 7):
        fieldnames.extend([f"j{i}_units", f"j{i}_deg"])

    try:
        time.sleep(0.5)
        origin_joints = read_joints(arm)
        origin_motor6 = read_motor6(arm)
        origin_motor6_pos = int(get_attr(origin_motor6, "pos", 0))
        joint_unwrapper = JointJ6Unwrapper(origin_joints[5])
        motor_unwrapper = LinearUnwrapper(origin_motor6_pos)

        logging.info("Start joint J6: %.3f deg (%d units)", deg(origin_joints[5]), origin_joints[5])
        logging.info("Start motor6 pos: %.3f deg (%d units)", deg(origin_motor6_pos), origin_motor6_pos)
        print("Recording feedback only. Rotate J6 left, then right, then back to center. Press Ctrl+C to stop.")
        print(f"Writing: {output}")

        period = 1.0 / args.hz
        next_tick = time.monotonic()
        t0 = next_tick
        sample = 0

        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            while True:
                now = time.monotonic()
                joints = read_joints(arm)
                high = arm.GetArmHighSpdInfoMsgs()
                motor6 = high.motor_6
                motor6_pos = int(get_attr(motor6, "pos", 0))

                j6_origin_delta, j6_step_delta, j6_unwrapped_delta = joint_unwrapper.update(joints[5])
                motor_origin_delta, motor_step_delta, motor_unwrapped_delta = motor_unwrapper.update(motor6_pos)

                motor_speed = int(get_attr(motor6, "motor_speed", 0))
                row = {
                    "sample": sample,
                    "time_s": f"{now - t0:.6f}",
                    "note": args.note,
                    "j6_units": joints[5],
                    "j6_deg": f"{deg(joints[5]):.6f}",
                    "j6_delta_from_start_units": j6_origin_delta,
                    "j6_delta_from_start_deg": f"{deg(j6_origin_delta):.6f}",
                    "j6_step_units": j6_step_delta,
                    "j6_step_deg": f"{deg(j6_step_delta):.6f}",
                    "j6_unwrapped_delta_units": j6_unwrapped_delta,
                    "j6_unwrapped_delta_deg": f"{deg(j6_unwrapped_delta):.6f}",
                    "motor6_pos_units": motor6_pos,
                    "motor6_pos_rad": f"{deg(motor6_pos):.6f}",
                    "motor6_delta_from_start_units": motor_origin_delta,
                    "motor6_delta_from_start_rad": f"{deg(motor_origin_delta):.6f}",
                    "motor6_step_units": motor_step_delta,
                    "motor6_step_rad": f"{deg(motor_step_delta):.6f}",
                    "motor6_unwrapped_delta_units": motor_unwrapped_delta,
                    "motor6_unwrapped_delta_rad": f"{deg(motor_unwrapped_delta):.6f}",
                    "motor6_speed_units": motor_speed,
                    "motor6_speed_rad_s": f"{deg(motor_speed):.6f}",
                    "motor6_current_raw": get_attr(motor6, "current", ""),
                    "motor6_effort_raw": get_attr(motor6, "effort", ""),
                    "motor6_can_id": get_attr(motor6, "can_id", ""),
                    "high_spd_hz": f"{float(get_attr(high, 'Hz', 0.0)):.3f}",
                }
                for i, value in enumerate(joints, start=1):
                    row[f"j{i}_units"] = value
                    row[f"j{i}_deg"] = f"{deg(value):.6f}"
                writer.writerow(row)

                sample += 1
                if sample % int(max(1, args.hz)) == 0:
                    f.flush()
                    print(
                        f"samples={sample} "
                        f"joint_j6={deg(joints[5]):.2f}deg "
                        f"motor6={rad(motor6_pos):.3f}rad/{rad_to_deg(rad(motor6_pos)):.1f}deg "
                        f"motor_delta={rad(motor_unwrapped_delta):.3f}rad/{rad_to_deg(rad(motor_unwrapped_delta)):.1f}deg"
                    )

                next_tick += period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print(f"\nStopped. Saved: {output}")
    finally:
        arm.DisconnectPort()


if __name__ == "__main__":
    main()
