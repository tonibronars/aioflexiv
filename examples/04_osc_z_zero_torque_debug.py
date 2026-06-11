#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from aioflexiv._loader import load_rt
from aioflexiv.config import config_path, resolve_robot_sn, save_last_robot_sn
from aioflexiv.controller import _damped_pinv, _rotation_error_vector
from aioflexiv.mujoco_model import MujocoModelBackend, default_mujoco_model_path


def _timestamp_seconds(timestamp) -> float:
    return float(timestamp[0]) + float(timestamp[1]) * 1e-9


def _rotvec_from_rotation(rotation: np.ndarray) -> np.ndarray:
    cos_angle = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    skew_vec = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-6:
        return 0.5 * skew_vec
    return skew_vec * (angle / (2.0 * np.sin(angle)))


def _as_list(value: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]


def _raw_array(raw, name: str, size: int) -> np.ndarray:
    value = getattr(raw, name, None)
    if value is None:
        return np.zeros(size)
    return np.asarray(value, dtype=float)


def _osc_probe(
    state: dict[str, np.ndarray],
    *,
    ee_goal: np.ndarray,
    q_goal: np.ndarray,
    z_kp: float,
    z_kd: float,
    null_kp: float,
    null_kd: float,
    damping: float,
    rcond: float,
) -> dict[str, object]:
    q = state["qpos"]
    dq = state["qvel"]
    link_dq = state.get("dq", dq)
    motor_dtheta = state.get("dtheta", dq)
    ee = state["ee"]
    jac = state["jac"]
    mm = state["mm"]

    twist_error = np.zeros(6)
    twist_error[:3] = ee_goal[:3, 3] - ee[:3, 3]
    twist_error[3:] = _rotation_error_vector(ee_goal[:3, :3], ee[:3, :3])
    ee_vel = jac @ dq

    ee_kp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, z_kp])
    ee_kd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, z_kd])
    task_cmd = ee_kp * twist_error - ee_kd * ee_vel
    ee_vel_from_dq = jac @ link_dq
    ee_vel_from_dtheta = jac @ motor_dtheta

    minv = np.linalg.pinv(mm)
    mx_inv = jac @ minv @ jac.T
    mx = _damped_pinv(mx_inv, damping=damping, rcond=rcond)
    old_mx = np.linalg.pinv(mx_inv)
    mm_sym_err = np.linalg.norm(mm - mm.T) / max(np.linalg.norm(mm), 1e-12)
    mm_eigs = np.linalg.eigvalsh(0.5 * (mm + mm.T))
    mx_sym_err = np.linalg.norm(mx - mx.T) / max(np.linalg.norm(mx), 1e-12)
    mx_col_z = mx[:, 5]
    mx_inv_col_z = mx_inv[:, 5]
    tau_per_unit_task_z = jac.T @ mx_col_z
    tau_per_unit_world_z_moment = jac.T @ np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    wrench = mx @ task_cmd
    old_wrench = old_mx @ task_cmd
    direct_wrench = task_cmd
    feedback = jac.T @ wrench
    old_feedback = jac.T @ old_wrench
    direct_feedback = jac.T @ direct_wrench

    jbar = minv @ jac.T @ mx
    null_projector = np.eye(jac.shape[1]) - jac.T @ jbar.T
    null_cmd = null_kp * (q_goal - q) - null_kd * dq
    null = null_projector @ null_cmd

    tau = feedback + null
    old_tau = old_feedback + null
    direct_tau = direct_feedback + null
    singular_values = np.linalg.svd(mx_inv, compute_uv=False)
    min_sv = float(singular_values[-1]) if singular_values.size else math.nan
    max_sv = float(singular_values[0]) if singular_values.size else math.nan
    cond = max_sv / min_sv if min_sv > 0.0 else math.inf

    return {
        "rot_err": _as_list(twist_error[3:]),
        "ee_vel": _as_list(ee_vel),
        "ee_vel_from_dq": _as_list(ee_vel_from_dq),
        "ee_vel_from_dtheta": _as_list(ee_vel_from_dtheta),
        "task_cmd": _as_list(task_cmd),
        "wrench": _as_list(wrench),
        "old_wrench": _as_list(old_wrench),
        "direct_wrench": _as_list(direct_wrench),
        "mx": _as_list(mx),
        "mx_inv": _as_list(mx_inv),
        "mx_col_z": _as_list(mx_col_z),
        "mx_inv_col_z": _as_list(mx_inv_col_z),
        "mx_zz": float(mx[5, 5]),
        "mx_inv_zz": float(mx_inv[5, 5]),
        "mx_col_z_linear_norm": float(np.linalg.norm(mx_col_z[:3])),
        "mx_col_z_xy_moment_norm": float(np.linalg.norm(mx_col_z[3:5])),
        "mx_sym_err": float(mx_sym_err),
        "mm_sym_err": float(mm_sym_err),
        "mm_eig_min": float(mm_eigs[0]),
        "mm_eig_max": float(mm_eigs[-1]),
        "mm_condition": float(mm_eigs[-1] / mm_eigs[0]) if mm_eigs[0] > 0 else math.inf,
        "tau_per_unit_task_z": _as_list(tau_per_unit_task_z),
        "tau_per_unit_task_z_norm": float(np.linalg.norm(tau_per_unit_task_z)),
        "tau_per_unit_task_z_max_abs": float(np.max(np.abs(tau_per_unit_task_z))),
        "tau_per_unit_world_z_moment": _as_list(tau_per_unit_world_z_moment),
        "tau_probe": _as_list(tau),
        "old_tau_probe": _as_list(old_tau),
        "direct_tau_probe": _as_list(direct_tau),
        "tau_probe_norm": float(np.linalg.norm(tau)),
        "tau_probe_max_abs": float(np.max(np.abs(tau))),
        "old_tau_probe_norm": float(np.linalg.norm(old_tau)),
        "old_tau_probe_max_abs": float(np.max(np.abs(old_tau))),
        "direct_tau_probe_norm": float(np.linalg.norm(direct_tau)),
        "direct_tau_probe_max_abs": float(np.max(np.abs(direct_tau))),
        "mx_inv_sv_min": min_sv,
        "mx_inv_sv_max": max_sv,
        "mx_inv_condition": cond,
        "jac_angular_z_row_norm": float(np.linalg.norm(jac[5, :])),
        "unit_world_z_moment_tau_norm": float(np.linalg.norm(tau_per_unit_world_z_moment)),
    }


def _read_full(
    ctrl,
    link_name: str,
    *,
    mujoco_backend: MujocoModelBackend | None = None,
    mujoco_velocity_source: str = "dtheta",
) -> dict[str, np.ndarray]:
    if mujoco_backend is not None:
        raw = ctrl.read_once(1000)
        qpos = np.asarray(raw.q, dtype=float)
        dq = np.asarray(raw.dq, dtype=float)
        dtheta = np.asarray(raw.dtheta, dtype=float)
        qvel = dtheta if mujoco_velocity_source == "dtheta" else dq
        state = {
            "timestamp": tuple(raw.timestamp),
            "qpos": qpos,
            "qvel": qvel,
            "dq": dq,
            "theta": np.asarray(raw.theta, dtype=float),
            "dtheta": dtheta,
            "tau": np.asarray(raw.tau, dtype=float),
            "tau_des": np.asarray(raw.tau_des, dtype=float),
            "tau_ext": np.asarray(raw.tau_ext, dtype=float),
            "tcp_vel": _raw_array(raw, "tcp_vel", 6),
            "model_backend": np.asarray(["mujoco"], dtype=object),
            "mujoco_velocity_source": np.asarray(
                [mujoco_velocity_source], dtype=object
            ),
        }
        state.update(mujoco_backend.state(qpos, qvel))
        return state

    raw = dict(ctrl.read_once_full(link_name, 1000))
    return {
        key: np.asarray(value, dtype=float)
        for key, value in raw.items()
        if key != "timestamp"
    } | {"timestamp": tuple(raw["timestamp"])}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream zero user torque and log z-orientation perturbation diagnostics "
            "for the OSC controller."
        )
    )
    parser.add_argument(
        "robot_sn",
        nargs="?",
        help=(
            "Robot serial number. If omitted, use the latest serial saved in "
            f"{config_path()}."
        ),
    )
    parser.add_argument("--link-name", default="flange")
    parser.add_argument("--model-backend", choices=("rdk", "mujoco"), default="rdk")
    parser.add_argument("--mujoco-model", default=None)
    parser.add_argument("--mujoco-body-name", default="link7")
    parser.add_argument("--mujoco-site-name", default=None)
    parser.add_argument(
        "--mujoco-velocity-source",
        choices=("dtheta", "dq"),
        default="dtheta",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--z-kp", type=float, default=128.0)
    parser.add_argument("--z-kd", type=float, default=32.0)
    parser.add_argument("--null-kp", type=float, default=0.0)
    parser.add_argument("--null-kd", type=float, default=0.0)
    parser.add_argument("--pinv-damping", type=float, default=1e-3)
    parser.add_argument("--pinv-rcond", type=float, default=1e-6)
    parser.add_argument("--friction-comp-scale", type=float, default=100.0)
    parser.add_argument("--disable-gravity-comp", action="store_true")
    parser.add_argument("--disable-soft-limits", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL output path. Defaults to osc_z_zero_torque_debug_<timestamp>.jsonl.",
    )
    args = parser.parse_args()

    try:
        robot_sn = resolve_robot_sn(args.robot_sn)
    except Exception as exc:
        parser.error(str(exc))

    out_path = args.out
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"osc_z_zero_torque_debug_{stamp}.jsonl")

    rt = load_rt()
    ctrl = None
    zero_torque: list[float] | None = None
    enable_gravity_comp = not args.disable_gravity_comp
    enable_soft_limits = not args.disable_soft_limits

    print(f"Connecting to {robot_sn}")
    print(
        "This streams zero user torque"
        f" (gravity_comp={enable_gravity_comp}, soft_limits={enable_soft_limits})."
    )
    print("Keep clear of the robot. Hold the arm still during baseline capture.")
    print("Starting in 3 seconds; press Ctrl+C to stop.")
    time.sleep(3.0)

    try:
        ctrl = rt.ActiveTorqueControl(robot_sn, [], False)
        save_last_robot_sn(robot_sn)
        mujoco_backend = None
        if args.model_backend == "mujoco":
            info = dict(ctrl.info())
            if args.mujoco_model is None and not info.get("model_name"):
                print(
                    "Warning: RobotInfo.model_name is unavailable; defaulting to "
                    "the Rizon4 MuJoCo model. Pass --mujoco-model to select "
                    "the XML explicitly."
                )
            model_path = (
                Path(args.mujoco_model).expanduser()
                if args.mujoco_model is not None
                else default_mujoco_model_path(str(info.get("model_name", "")))
            )
            mujoco_backend = MujocoModelBackend(
                model_path,
                site_name=args.mujoco_site_name,
                body_name=args.mujoco_body_name,
            )
            print(
                "Using MuJoCo model "
                f"{mujoco_backend.path} "
                f"({mujoco_backend.frame_type}={mujoco_backend.frame_name}, "
                f"qvel={args.mujoco_velocity_source})"
            )

        first = _read_full(
            ctrl,
            args.link_name,
            mujoco_backend=mujoco_backend,
            mujoco_velocity_source=args.mujoco_velocity_source,
        )
        zero_torque = [0.0] * int(np.asarray(first["qpos"]).shape[0])
        settle_deadline = time.perf_counter() + max(args.settle, 0.0)
        while time.perf_counter() < settle_deadline:
            ctrl.write_once(
                zero_torque,
                enable_gravity_comp,
                enable_soft_limits,
                args.friction_comp_scale,
            )
            ctrl.read_once(1000)

        baseline = _read_full(
            ctrl,
            args.link_name,
            mujoco_backend=mujoco_backend,
            mujoco_velocity_source=args.mujoco_velocity_source,
        )
        ee_goal = np.asarray(baseline["ee"], dtype=float).copy()
        q_goal = np.asarray(baseline["qpos"], dtype=float).copy()
        start_robot_t = _timestamp_seconds(baseline["timestamp"])
        start_wall_t = time.perf_counter()
        sample_period = 1.0 / args.sample_hz if args.sample_hz > 0 else 0.0
        print_period = 1.0 / args.print_hz if args.print_hz > 0 else math.inf
        next_sample_t = start_wall_t
        next_print_t = start_wall_t
        deadline = start_wall_t + args.duration if args.duration > 0 else math.inf
        previous_ee = ee_goal.copy()
        previous_robot_t = start_robot_t
        samples = 0
        stream_iterations = 0

        print(f"Baseline captured. Perturb z orientation now. Logging to {out_path}")
        with out_path.open("w", encoding="utf-8") as output:
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                state = _read_full(
                    ctrl,
                    args.link_name,
                    mujoco_backend=mujoco_backend,
                    mujoco_velocity_source=args.mujoco_velocity_source,
                )
                ctrl.write_once(
                    zero_torque,
                    enable_gravity_comp,
                    enable_soft_limits,
                    args.friction_comp_scale,
                )
                stream_iterations += 1

                robot_t = _timestamp_seconds(state["timestamp"])
                dt = max(robot_t - previous_robot_t, 1e-9)
                ee = np.asarray(state["ee"], dtype=float)

                if now >= next_sample_t:
                    rotation_delta = ee[:3, :3] @ previous_ee[:3, :3].T
                    fd_omega = _rotvec_from_rotation(rotation_delta) / dt
                    probe = _osc_probe(
                        state,
                        ee_goal=ee_goal,
                        q_goal=q_goal,
                        z_kp=args.z_kp,
                        z_kd=args.z_kd,
                        null_kp=args.null_kp,
                        null_kd=args.null_kd,
                        damping=args.pinv_damping,
                        rcond=args.pinv_rcond,
                    )

                    ee_vel = np.asarray(probe["ee_vel"], dtype=float)
                    ee_vel_from_dq = np.asarray(probe["ee_vel_from_dq"], dtype=float)
                    ee_vel_from_dtheta = np.asarray(
                        probe["ee_vel_from_dtheta"], dtype=float
                    )
                    rot_err = np.asarray(probe["rot_err"], dtype=float)
                    record = {
                        "t_s": robot_t - start_robot_t,
                        "wall_t_s": time.perf_counter() - start_wall_t,
                        "stream_iterations": stream_iterations,
                        "q": _as_list(state["qpos"]),
                        "qvel": _as_list(state["qvel"]),
                        "dq": _as_list(state["dq"]),
                        "dtheta": _as_list(
                            state.get("dtheta", np.zeros_like(state["dq"]))
                        ),
                        "tau": _as_list(state["tau"]),
                        "tau_des": _as_list(state["tau_des"]),
                        "tau_ext": _as_list(state["tau_ext"]),
                        "position": _as_list(ee[:3, 3]),
                        "rot_err": probe["rot_err"],
                        "rot_err_norm": float(np.linalg.norm(rot_err)),
                        "rot_err_z_rad": float(rot_err[2]),
                        "omega_jac": _as_list(ee_vel[3:]),
                        "omega_jac_z": float(ee_vel[5]),
                        "omega_jac_from_dq": _as_list(ee_vel_from_dq[3:]),
                        "omega_jac_from_dq_z": float(ee_vel_from_dq[5]),
                        "omega_jac_from_dtheta": _as_list(ee_vel_from_dtheta[3:]),
                        "omega_jac_from_dtheta_z": float(ee_vel_from_dtheta[5]),
                        "omega_fd": _as_list(fd_omega),
                        "omega_fd_z": float(fd_omega[2]),
                        "omega_tcp": _as_list(state.get("tcp_vel", np.zeros(6))[3:]),
                        "omega_tcp_z": float(state.get("tcp_vel", np.zeros(6))[5]),
                        "omega_z_jac_minus_fd": float(ee_vel[5] - fd_omega[2]),
                        "omega_z_dq_minus_fd": float(ee_vel_from_dq[5] - fd_omega[2]),
                        "omega_z_dtheta_minus_fd": float(
                            ee_vel_from_dtheta[5] - fd_omega[2]
                        ),
                    } | probe

                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    samples += 1

                    if now >= next_print_t:
                        print(
                            "t={t:6.2f}s err_z={err:+.4f} "
                            "wz_qvel={wz:+.4f} wz_dq={wdq:+.4f} wz_fd={wf:+.4f} "
                            "tau_probe_max={tm:.2f} direct_tau_max={dtm:.2f} "
                            "mxzz={mxzz:.4g} tau/task_z={ttz:.4g} cond={cond:.2e}".format(
                                t=record["t_s"],
                                err=record["rot_err_z_rad"],
                                wz=record["omega_jac_z"],
                                wdq=record["omega_jac_from_dq_z"],
                                wf=record["omega_fd_z"],
                                tm=record["tau_probe_max_abs"],
                                dtm=record["direct_tau_probe_max_abs"],
                                mxzz=record["mx_zz"],
                                ttz=record["tau_per_unit_task_z_max_abs"],
                                cond=record["mx_inv_condition"],
                            )
                        )
                        next_print_t += print_period

                    if sample_period > 0.0:
                        while next_sample_t <= now:
                            next_sample_t += sample_period

                previous_ee = ee.copy()
                previous_robot_t = robot_t

        print(
            f"Done. Wrote {samples} samples to {out_path} "
            f"over {stream_iterations} zero-torque stream iterations."
        )
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping robot...")
    finally:
        if ctrl is not None:
            ctrl.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
