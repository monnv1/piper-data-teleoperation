"""Set Piper joint zero using the official JointConfig command.

Default target is joint 6. This script follows the SDK demo pattern:
1. connect to CAN,
2. optionally disable the selected joint motor,
3. wait for the operator to manually place the joint at the desired zero,
4. send JointConfig(joint, set_zero=0xAE),
5. optionally re-enable the selected motor.

It does not send JointCtrl or move the arm to any pose.
"""
from __future__ import annotations

import argparse
import time

from teleoperation.teleop import FOLLOWER_CAN, make_piper, read_joints, read_motor6_pos_mrad

CONFIRM_TEXT = "SET J6 ZERO"


def units_to_deg(units: int) -> float:
    return units / 1000.0


def print_feedback(piper, label: str) -> None:
    joints = read_joints(piper)
    motor6_mrad = read_motor6_pos_mrad(piper)
    print(f"{label} joint deg: {[round(units_to_deg(x), 3) for x in joints]}")
    print(f"{label} joint6={units_to_deg(joints[5]):.3f} deg, motor6={motor6_mrad / 1000.0:.6f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set Piper joint zero with JointConfig(set_zero=0xAE)")
    parser.add_argument("--can", default=FOLLOWER_CAN, help="CAN interface, default: can0")
    parser.add_argument("--joint", type=int, default=6, choices=range(1, 7), help="Joint number to zero, default: 6")
    parser.add_argument("--no-disable", action="store_true", help="Do not disable the selected joint before zeroing")
    parser.add_argument("--no-reenable", action="store_true", help="Do not re-enable the selected joint after zeroing")
    parser.add_argument("--repeat", type=int, default=3, help="Number of JointConfig zero commands to send")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between repeated zero commands")
    parser.add_argument("--yes", action="store_true", help="Skip typed confirmation; still waits for ENTER before zeroing")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.interval < 0:
        raise ValueError("--interval must be non-negative")

    piper = make_piper(args.can)
    piper.ConnectPort()
    try:
        time.sleep(0.5)
        print_feedback(piper, "Before")
        print()
        print(f"This will set joint {args.joint} current physical position as its new zero on {args.can}.")
        print("Support the arm and keep the emergency stop reachable. Do not run teleop/deploy at the same time.")
        if not args.yes:
            typed = input(f"Type exactly '{CONFIRM_TEXT}' to continue: ").strip()
            if typed != CONFIRM_TEXT:
                raise RuntimeError("Confirmation text did not match; cancelled")

        if not args.no_disable:
            print(f"Disabling joint {args.joint} motor...")
            piper.DisableArm(args.joint)
            time.sleep(0.5)

        print(f"Manually place joint {args.joint} at the desired ZERO position, then press ENTER.")
        input()
        print_feedback(piper, "At zero pose")

        for i in range(args.repeat):
            print(f"Sending JointConfig(joint_num={args.joint}, set_zero=0xAE) [{i + 1}/{args.repeat}]...")
            piper.JointConfig(args.joint, 0xAE)
            time.sleep(args.interval)

        if not args.no_reenable:
            print(f"Re-enabling joint {args.joint} motor...")
            piper.EnableArm(args.joint)
            time.sleep(0.5)

        print_feedback(piper, "After")
        print("Done. Power-cycle/reconnect if feedback does not refresh immediately.")
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
