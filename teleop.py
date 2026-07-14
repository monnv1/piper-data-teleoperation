"""Soft teleoperation: leader Piper → follower Piper via CAN.

基于 AgileX Piper SDK V2 官方文档:
  - https://github.com/agilexrobotics/piper_sdk
  - double_piper.MD / INTERFACE_V2.MD

关键流程:
  1. 连接两台机械臂
  2. 使能主臂 (EnablePiper) — 重力补偿生效，可自由拖拽
  3. 从臂: 读当前关节 → MotionCtrl_1(RESET) 清除旧目标 → 发保持指令 → 使能
  4. 进入跟随循环 (高跟随模式 0xAD)
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any

LEADER_CAN = "can1"
FOLLOWER_CAN = "can0"

ALPHA = 0.2
MAX_STEP_UNITS = 2000
GRIPPER_ALPHA = 0.8
GRIPPER_MAX_STEP_UNITS = 3000
LOOP_HZ = 50.0
STATUS_PRINT_INTERVAL = 1.0
JOINT_LIMITS_UNITS = [
    (-150000, 150000),
    (0, 180000),
    (-170000, 0),
    (-100000, 100000),
    (-70000, 70000),
    (-120000, 120000),
]
# 关节偏移 (SDK 单位: 0.001°)。J6 硬件零点已重新设置，默认不再加任何偏置。
JOINT_OFFSET = [0, 0, 0, 0, 0, 0]


def make_piper(can_name: str):
    from piper_sdk import C_PiperInterface_V2

    try:
        return C_PiperInterface_V2(
            can_name=can_name,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
        )
    except TypeError:
        return C_PiperInterface_V2(can_name)


def extract_joints(msg: Any) -> list[int]:
    stack: list[Any] = [msg]
    seen: set[int] = set()
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if all(hasattr(x, n) for n in joint_names):
            return [int(getattr(x, n)) for n in joint_names]
        if isinstance(x, (list, tuple)):
            stack.extend(x)
        for k in ["joint_state", "joint_states", "arm_joint", "arm_joint_msgs"]:
            if hasattr(x, k):
                stack.append(getattr(x, k))
    raise RuntimeError("Could not parse joint state from Piper SDK message")


def read_joints(piper) -> list[int]:
    return extract_joints(piper.GetArmJointMsgs())


def read_gripper(piper) -> int | None:
    try:
        msg = piper.GetArmGripperMsgs()
        return int(msg.gripper_state.grippers_angle)
    except Exception:
        return None


def clamp_joint_units(value: int, joint_index: int) -> int:
    lo, hi = JOINT_LIMITS_UNITS[joint_index]
    return max(lo, min(value, hi))


def read_motor6_pos_mrad(piper) -> int:
    return int(piper.GetArmHighSpdInfoMsgs().motor_6.pos)


def map_follower_target(
    leader_units: list[int],
    joint_offset: list[int] | None,
) -> list[int]:
    target = []
    for i in range(6):
        v = leader_units[i] + joint_offset[i] if joint_offset else leader_units[i]
        target.append(clamp_joint_units(v, i))
    return target


def run_teleop(
    leader_can: str,
    follower_can: str,
    alpha: float = ALPHA,
    max_step_units: int = MAX_STEP_UNITS,
    loop_hz: float = LOOP_HZ,
    command_gripper: bool = True,
    gripper_effort: int = 1000,
    gripper_alpha: float = GRIPPER_ALPHA,
    gripper_max_step_units: int = GRIPPER_MAX_STEP_UNITS,
    speed_percent: int = 10,
    joint_offset: list[int] | None = None,
    do_home: bool = False,
) -> None:
    from piper_sdk import C_PiperInterface_V2

    # ===== 1. 连接 =========================================================
    logging.info("Connecting leader on %s …", leader_can)
    leader = make_piper(leader_can)
    leader.ConnectPort()
    time.sleep(0.5)

    logging.info("Connecting follower on %s …", follower_can)
    follower = make_piper(follower_can)
    follower.ConnectPort()
    time.sleep(0.5)

    # ===== 2. 配置主从模式 ==================================================
    # 从臂模式 (0xFC): 上报关节反馈 + 响应 JointCtrl
    # 主臂模式 (0xFA): 不报关节反馈 + 发控制指令 (这里我们不需要)
    # 两台都设从臂模式，因为我们用 Python 软接力，不是硬 CAN 直连
    logging.info("Configuring both arms as slave mode …")
    leader.MasterSlaveConfig(0xFC, 0, 0, 0)
    follower.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.2)

    # ===== 3. 使能主臂 (重力补偿) ==========================================
    logging.info("Enabling leader (gravity compensation) …")
    if not leader.EnablePiper():
        logging.warning("Leader EnablePiper returned False, retrying …")
        time.sleep(0.5)
        leader.EnablePiper()
    # 切 Standby 模式: 电机使能 + 重力补偿, 无位置锁定, 可自由拖拽
    leader.MotionCtrl_2(0x00, 0x01, 1, 0x00)
    time.sleep(0.3)
    logging.info("Leader ready — drag freely.")

    # ===== 4. 从臂: 使能 → 保持 ============================================
    # 先读当前关节, 使能后立刻发 JointCtrl 保持, 避免电机咬合时抖动
    # (不要 MotionCtrl_1 复位, 那不是常规上电流程)
    leader_origin = read_joints(leader)
    home = read_joints(follower)
    logging.info("Leader origin [deg]: %s", [round(u / 1000.0, 2) for u in leader_origin])
    logging.info("Follower home [deg]: %s", [round(u / 1000.0, 2) for u in home])

    logging.info("Enabling follower …")
    if not follower.EnablePiper():
        logging.warning("Follower EnablePiper returned False, retrying …")
        time.sleep(0.5)
        follower.EnablePiper()
    time.sleep(0.2)

    # 使能后立刻发 JointCtrl 保持当前位置
    follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    follower.JointCtrl(*home)
    time.sleep(0.2)

    if command_gripper:
        g0 = read_gripper(leader)
        if g0 is not None:
            # Official V2: 0x02 clears while disabled, 0x03 enables and clears errors.
            follower.GripperCtrl(g0, gripper_effort, 0x02, 0)
            time.sleep(0.05)
            follower.GripperCtrl(g0, gripper_effort, 0x03, 0)
            time.sleep(0.15)
        else:
            logging.warning("Leader gripper feedback unavailable at startup.")

    # ===== 5. 进入高跟随模式 (0xAD) ========================================
    # 0xAD = MIT 高速跟随模式，延迟最低 (double_piper.MD §2.2)
    follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
    logging.info(
        "Teleoperation started (Ctrl+C to stop) — speed=%d%%, alpha=%.1f",
        speed_percent, alpha,
    )

    last_filtered: list[int] | None = None
    last_gripper_val: int | None = None
    period = 1.0 / loop_hz
    next_tick = time.monotonic()
    last_status = 0.0

    try:
        while True:
            now = time.monotonic()

            # --- 读主臂关节 -------------------------------------------------
            raw = read_joints(leader)
            if last_filtered is None:
                last_filtered = raw[:]

            # --- 低通 + 限速 ------------------------------------------------
            q_filt: list[int] = []
            for i in range(6):
                v = int(last_filtered[i] * (1.0 - alpha) + raw[i] * alpha)
                dv = v - last_filtered[i]
                dv = max(-max_step_units, min(max_step_units, dv))
                q_filt.append(last_filtered[i] + dv)
            last_filtered = q_filt

            # --- 映射到从臂目标并做官方关节限位保护 ---------------------------
            q_target = map_follower_target(q_filt, joint_offset)
            follower.MotionCtrl_2(0x01, 0x01, speed_percent, 0xAD)
            follower.JointCtrl(*q_target)

            # --- 夹爪 -------------------------------------------------------
            if command_gripper:
                g = read_gripper(leader)
                if g is not None:
                    if last_gripper_val is None:
                        last_gripper_val = g
                    gf = int(last_gripper_val * (1.0 - gripper_alpha) + g * gripper_alpha)
                    dg = gf - last_gripper_val
                    dg = max(-gripper_max_step_units, min(gripper_max_step_units, dg))
                    gf = last_gripper_val + dg
                    last_gripper_val = gf
                    follower.GripperCtrl(gf, gripper_effort, 0x01, 0)

            # --- 状态输出 ---------------------------------------------------
            if now - last_status >= STATUS_PRINT_INTERVAL:
                deg = [round(u / 1000.0, 2) for u in q_target]
                extra = ""
                if command_gripper and last_gripper_val is not None:
                    extra = f"  gripper={last_gripper_val / 1000.0:.1f} mm"
                logging.info("follower [deg]: %s%s", deg, extra)
                last_status = now

            # --- 定周期 -----------------------------------------------------
            next_tick += period
            sleep_seconds = next_tick - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        logging.info("Stopped by user")
    finally:
        if do_home:
            home_units = [int(d * 1000) for d in HOME_DEG]
            logging.info("Moving follower to home %s …", HOME_DEG)
            _move_to_home(follower, home_units, HOME_SPEED)
            logging.info("Moving leader to home %s …", HOME_DEG)
            leader.MotionCtrl_2(0x01, 0x01, HOME_SPEED, 0x00)
            _move_to_home(leader, home_units, HOME_SPEED)
            time.sleep(0.2)
        else:
            try:
                hold = read_joints(follower)
            except Exception:
                hold = last_filtered
            if hold is not None:
                follower.MotionCtrl_2(0x01, 0x01, 1, 0x00)
                follower.JointCtrl(*hold)
                time.sleep(0.2)
            if command_gripper:
                hold_gripper = read_gripper(follower)
                if hold_gripper is not None:
                    follower.GripperCtrl(hold_gripper, gripper_effort, 0x01, 0)
                    time.sleep(0.1)
        follower.DisconnectPort()
        leader.DisconnectPort()
        logging.info("Disconnected.")


HOME_DEG = [0.0, 90.0, -90.0, 0.0, 69.0, 0.0]  # 回到仿真初始构型
HOME_SPEED = 3                                   # 回零速度 %


def _move_to_home(piper, joints_units: list[int], speed: int) -> None:
    """慢速 MOVE J 回到目标关节角, 等待稳定."""
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    deadline = time.monotonic() + 30.0
    settled = 0
    while time.monotonic() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*joints_units)
        current = read_joints(piper)
        err = max(abs(current[i] - joints_units[i]) for i in range(6))
        if err <= 1500:  # 1.5°
            settled += 1
            if settled >= 5:
                break
        else:
            settled = 0
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soft teleoperation — leader arm drives follower arm via CAN"
    )
    parser.add_argument("--leader-can", default=LEADER_CAN)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-step", type=int, default=MAX_STEP_UNITS)
    parser.add_argument("--hz", type=float, default=LOOP_HZ)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-effort", type=int, default=1000, help="Gripper torque 0.001 N/m (default: 1000, max: 5000)")
    parser.add_argument("--gripper-alpha", type=float, default=GRIPPER_ALPHA)
    parser.add_argument("--gripper-max-step", type=int, default=GRIPPER_MAX_STEP_UNITS,
                        help="Max gripper change per cycle, in 0.001 mm units")
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--offset", type=int, nargs=6, default=JOINT_OFFSET,
                        help=f"Joint offset 0.001° units (default: {JOINT_OFFSET})")
    parser.add_argument("--home", action="store_true",
                        help="Stop后自动回零（所有轴回到仿真初始构型）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not 1 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be in [1, 100]")

    run_teleop(
        leader_can=args.leader_can,
        follower_can=args.follower_can,
        alpha=args.alpha,
        max_step_units=args.max_step,
        loop_hz=args.hz,
        command_gripper=not args.no_gripper,
        gripper_effort=args.gripper_effort,
        gripper_alpha=args.gripper_alpha,
        gripper_max_step_units=args.gripper_max_step,
        speed_percent=args.speed_percent,
        joint_offset=args.offset,
        do_home=args.home,
    )


if __name__ == "__main__":
    main()
