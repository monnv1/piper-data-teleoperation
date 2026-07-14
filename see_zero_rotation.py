"""离线 IK 求解：分别令 RX / RY / RZ = 0，显示三种情况的关节解。"""
from __future__ import annotations

import numpy as np

from deploy.kinematics.piper_ik import PiperHostIK


def main() -> None:
    from piper_sdk import C_PiperForwardKinematics

    fk = C_PiperForwardKinematics(1)
    ik = PiperHostIK(
        fk=fk,
        position_tolerance_m=0.005,
        rotation_tolerance_rad=0.03,
        max_joint_step_deg=90.0,
        min_joint_limit_margin_deg=0.5,
        max_nfev=100,
    )

    # 有三个起始构型，每个有一个轴显著非零，另两个接近 0
    scenarios = [
        ("RX=0 (起始 RX=35°, RY/RZ≈0)", [0.0, 30.0, -100.0, 35.0, 10.0, 0.0]),
        ("RY=0 (起始 RY=35°, RX/RZ≈0)", [0.0, 65.0, -90.0, 0.0, 20.0, 0.0]),
        ("RZ=0 (起始 RZ=30°, RX/RY≈0)", [30.0, 40.0, -90.0, 0.0, 10.0, 30.0]),
    ]

    for desc, j0 in scenarios:
        rad = np.radians(j0)
        pose = np.asarray(fk.CalFK(rad.tolist())[-1], dtype=np.float64)
        pos_m = pose[:3] / 1000.0
        euler_deg = pose[3:]
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
        print(f"  Start joints (deg): {j0}")
        print(f"  Position (mm):      {np.round(pos_m * 1000, 1).tolist()}")
        print(f"  Euler XYZ (deg):    RX={euler_deg[0]:.1f}  RY={euler_deg[1]:.1f}  RZ={euler_deg[2]:.1f}")

        euler_rad = np.radians(euler_deg)

        def try_solve(label: str, target_euler_rad):
            target_deg = np.degrees(target_euler_rad)
            try:
                result = ik.solve(pos_m, target_euler_rad, rad)
                q_deg = result["selected_joint_degrees"]
                fk_out = np.asarray(fk.CalFK(np.radians(q_deg).tolist())[-1], dtype=np.float64)
                fk_pos = fk_out[:3] / 1000.0
                fk_euler = fk_out[3:]
                pos_err = result["selected"]["position_error_m"]
                rot_err = result["selected"]["rotation_error_rad"]
                print(f"  → {label}  joints (deg): {np.round(q_deg, 1).tolist()}")
                print(f"     FK pose (mm):      {np.round(fk_pos * 1000, 1).tolist()}")
                print(f"     FK Euler XYZ (deg): RX={fk_euler[0]:.1f}  RY={fk_euler[1]:.1f}  RZ={fk_euler[2]:.1f}")
                print(f"     pos_err={pos_err:.4f}m  rot_err={rot_err:.4f}rad")
            except Exception as e:
                print(f"  → {label}  FAILED: {e}")

        try_solve("RX=0", np.array([0.0, euler_rad[1], euler_rad[2]]))
        try_solve("RY=0", np.array([euler_rad[0], 0.0, euler_rad[2]]))
        try_solve("RZ=0", np.array([euler_rad[0], euler_rad[1], 0.0]))


if __name__ == "__main__":
    main()
