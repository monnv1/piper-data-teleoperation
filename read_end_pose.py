"""以 0.5s 间隔打印机械臂末端位姿 (X Y Z RX RY RZ)。"""
from __future__ import annotations

import argparse
import time


def _int16(v: int) -> int:
    return v - 65536 if v >= 32768 else v


def main() -> None:
    parser = argparse.ArgumentParser(description="Print end pose at 2 Hz")
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--hz", type=float, default=2.0, help="Print frequency (default: 2)")
    args = parser.parse_args()

    period = 1.0 / args.hz

    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can_name=args.can)
    piper.ConnectPort()
    time.sleep(0.5)

    piper.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.2)

    for attempt in range(3):
        if piper.EnablePiper():
            break
        print(f"EnablePiper attempt {attempt + 1} failed, retrying…")
        time.sleep(0.5)
    else:
        print("Warning: EnablePiper did not return True")

    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    time.sleep(0.3)

    factor_mm = 0.001
    factor_deg = 0.001
    print(f"{'X (mm)':>10} {'Y (mm)':>10} {'Z (mm)':>10} {'RX (deg)':>10} {'RY (deg)':>10} {'RZ (deg)':>10}")
    print("-" * 66)

    next_tick = time.monotonic()
    try:
        while True:
            pose = piper.GetArmEndPoseMsgs().end_pose
            x = _int16(pose.X_axis) * factor_mm
            y = _int16(pose.Y_axis) * factor_mm
            z = _int16(pose.Z_axis) * factor_mm
            rx = _int16(pose.RX_axis) * factor_deg
            ry = _int16(pose.RY_axis) * factor_deg
            rz = _int16(pose.RZ_axis) * factor_deg
            print(f"{x:10.2f} {y:10.2f} {z:10.2f} {rx:10.2f} {ry:10.2f} {rz:10.2f}", flush=True)

            next_tick += period
            sleep_seconds = next_tick - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
