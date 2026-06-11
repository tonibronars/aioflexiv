#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from aioflexiv import FlexivController


def _rotation_error_vector(R_goal: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    R_err = R_goal @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    skew_vec = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    if angle < 1e-6:
        return 0.5 * skew_vec
    return skew_vec * (angle / (2.0 * np.sin(angle)))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_sn", nargs="?")
    parser.add_argument(
        "--model-backend",
        choices=("rdk", "mujoco"),
        default="mujoco",
        help="Source for OSC kinematics/dynamics. Defaults to MuJoCo for this example.",
    )
    parser.add_argument(
        "--mujoco-model",
        default=None,
        help="Path to a MuJoCo XML model. Defaults from RobotInfo.model_name.",
    )
    parser.add_argument("--mujoco-body-name", default="link7")
    parser.add_argument("--mujoco-site-name", default=None)
    parser.add_argument(
        "--mujoco-velocity-source",
        choices=("dtheta", "dq"),
        default="dtheta",
        help="Joint velocity used to advance MuJoCo and compute OSC damping.",
    )
    args = parser.parse_args()

    controller = FlexivController(
        args.robot_sn,
        model_backend=args.model_backend,
        mujoco_model_path=args.mujoco_model,
        mujoco_body_name=args.mujoco_body_name,
        mujoco_site_name=args.mujoco_site_name,
        mujoco_velocity_source=args.mujoco_velocity_source,
    )
    await controller.start()
    try:
        state = controller.state
        backend = state.get("model_backend", np.asarray(["rdk"], dtype=object))[0]
        print(f"OSC model backend: {backend}")
        if backend == "mujoco":
            print(
                "MuJoCo model: "
                f"{state['mujoco_model_path'][0]} "
                f"({state['mujoco_frame_type'][0]}={state['mujoco_frame_name'][0]}, "
                f"qvel={state['mujoco_velocity_source'][0]})"
            )
        base = np.ones(controller.dof)
        with controller.state_lock:
            controller.kp = base * 80.0
            controller.kd = base * 4.0
            print("Moving to initial position...")
        await controller.move()

        controller.switch("osc")
        controller.ee_kp = np.array([64.0, 64.0, 64.0, 128.0, 128.0, 128.0])
        controller.ee_kd = np.array([32.0, 32.0, 32.0, 32.0, 32.0, 4.0])
        controller.null_kp = np.ones(controller.dof) * 0.0
        controller.null_kd = np.ones(controller.dof) * 0.0
        controller.set_freq(50)

        target = controller.initial_ee.copy()
        last_print = time.perf_counter()
        while True:
            await controller.set("ee_desired", target)
            now = time.perf_counter()
            if now - last_print >= 1.0 and controller.state is not None:
                state = controller.state
                raw_tau = state.get("controller_torque")
                streamed_tau = state.get("last_torque")
                ee = state.get("ee")
                if raw_tau is not None and streamed_tau is not None and ee is not None:
                    pos_err = np.linalg.norm(target[:3, 3] - ee[:3, 3])
                    rot_err = _rotation_error_vector(target[:3, :3], ee[:3, :3])
                    limit = controller.torque_limit
                    clipped = (
                        limit is not None
                        and np.any(
                            np.isclose(
                                np.abs(streamed_tau),
                                limit,
                                rtol=0.0,
                                atol=1e-6,
                            )
                        )
                    )
                    print(
                        "osc "
                        f"pos_err={pos_err:.4f}m "
                        f"rot_err={np.linalg.norm(rot_err):.4f}rad "
                        f"rot_z={rot_err[2]:+.4f}rad "
                        f"raw_tau_max={np.max(np.abs(raw_tau)):.2f}Nm "
                        f"sent_tau_max={np.max(np.abs(streamed_tau)):.2f}Nm "
                        f"limit_clipped={clipped}"
                    )
                last_print = now
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
