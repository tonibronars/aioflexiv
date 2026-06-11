from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from ._loader import load_rt
from .config import resolve_robot_sn, save_last_robot_sn
from .mujoco_model import MujocoModelBackend, default_mujoco_model_path
from .tools import switch_active_tool


class FlexivRobotInterface:
    """Low-level bridge for Flexiv RT joint-torque control.

    This class owns the C++ `ActiveTorqueControl` shim. It does not use
    `flexiv::rdk::Scheduler`; timing is driven by waiting for fresh robot state
    timestamps and immediately writing the next torque command from Python.
    """

    def __init__(
        self,
        robot_sn: str | None = None,
        *,
        link_name: str = "flange",
        tool: str | None = None,
        enable_gravity_comp: bool = True,
        enable_soft_limits: bool = True,
        friction_comp_scale: float = 100.0,
        network_interface_whitelist: list[str] | None = None,
        verbose: bool = False,
        auto_clear_fault: bool = True,
        fault_clear_timeout_sec: int = 30,
        operational_timeout_sec: int = 30,
        model_backend: str = "rdk",
        mujoco_model_path: str | Path | None = None,
        mujoco_site_name: str | None = None,
        mujoco_body_name: str = "link7",
        mujoco_velocity_source: str = "dtheta",
    ) -> None:
        self.robot_sn = resolve_robot_sn(robot_sn)
        self.link_name = link_name
        self.tool = tool
        self.enable_gravity_comp = enable_gravity_comp
        self.enable_soft_limits = enable_soft_limits
        self.friction_comp_scale = friction_comp_scale
        self.network_interface_whitelist = network_interface_whitelist or []
        self.verbose = verbose
        self.auto_clear_fault = bool(auto_clear_fault)
        self.fault_clear_timeout_sec = int(fault_clear_timeout_sec)
        self.operational_timeout_sec = int(operational_timeout_sec)
        self.model_backend = str(model_backend).lower()
        if self.model_backend not in {"rdk", "mujoco"}:
            raise ValueError("model_backend must be 'rdk' or 'mujoco'")
        self.mujoco_model_path = (
            None
            if mujoco_model_path is None
            else Path(mujoco_model_path).expanduser()
        )
        self.mujoco_site_name = mujoco_site_name
        self.mujoco_body_name = mujoco_body_name
        self.mujoco_velocity_source = str(mujoco_velocity_source).lower()
        if self.mujoco_velocity_source not in {"dq", "dtheta"}:
            raise ValueError("mujoco_velocity_source must be 'dq' or 'dtheta'")

        self._rt = None
        self._ctrl = None
        self._info: dict[str, Any] | None = None
        self._last_torque: np.ndarray | None = None
        self._mujoco_backend: MujocoModelBackend | None = None

    @property
    def started(self) -> bool:
        return self._ctrl is not None

    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            raise RuntimeError("Robot interface has not been started")
        return self._info

    @property
    def dof(self) -> int:
        return int(self.info["dof"])

    @property
    def torque_limit(self) -> np.ndarray:
        tau_max = np.asarray(self.info.get("tau_max", []), dtype=float)
        if (
            tau_max.shape == (self.dof,)
            and np.all(np.isfinite(tau_max))
            and np.all(tau_max > 0)
        ):
            return tau_max.copy()
        raise RuntimeError(
            "Flexiv RDK did not report valid RobotInfo.tau_max torque limits; "
            "refusing to use guessed fallback limits."
        )

    def start(self) -> None:
        if self._ctrl is not None:
            return
        if self.tool:
            switch_active_tool(
                self.robot_sn,
                self.tool,
                network_interface_whitelist=self.network_interface_whitelist,
                verbose=self.verbose,
            )
        self._rt = load_rt()
        self._ctrl = self._rt.ActiveTorqueControl(
            self.robot_sn,
            self.network_interface_whitelist,
            self.verbose,
            self.auto_clear_fault,
            self.fault_clear_timeout_sec,
            self.operational_timeout_sec,
        )
        self._info = dict(self._ctrl.info())
        self._initialize_model_backend()
        try:
            save_last_robot_sn(self.robot_sn)
        except Exception as exc:
            warnings.warn(
                f"Could not save last robot serial to aioflexiv config: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._last_torque = np.zeros(self.dof)

    def stop(self) -> None:
        if self._ctrl is not None:
            self._ctrl.stop()
            self._ctrl = None
            self._mujoco_backend = None

    def _require_started(self):
        if self._ctrl is None:
            raise RuntimeError("Robot interface has not been started")
        return self._ctrl

    def _initialize_model_backend(self) -> None:
        self._mujoco_backend = None
        if self.model_backend != "mujoco":
            return
        if self._info is None:
            raise RuntimeError("Robot info must be available before model backend setup")
        model_name = self._info.get("model_name")
        if self.mujoco_model_path is None and not model_name:
            warnings.warn(
                "Flexiv RDK RobotInfo.model_name is unavailable; defaulting to the "
                "Rizon4 MuJoCo model. Rebuild the RT extension or pass "
                "mujoco_model_path for explicit model selection.",
                RuntimeWarning,
                stacklevel=2,
            )
        model_path = (
            self.mujoco_model_path
            if self.mujoco_model_path is not None
            else default_mujoco_model_path(str(model_name or ""))
        )
        self._mujoco_backend = MujocoModelBackend(
            model_path,
            site_name=self.mujoco_site_name,
            body_name=self.mujoco_body_name,
        )

    def _last_or_raw_torque(self, raw) -> np.ndarray:
        return (
            self._last_torque.copy()
            if self._last_torque is not None
            else np.asarray(raw.tau_des, dtype=float)
        )

    @staticmethod
    def _raw_array(raw, name: str, size: int) -> np.ndarray:
        value = getattr(raw, name, None)
        if value is None:
            return np.zeros(size)
        return np.asarray(value, dtype=float)

    def state_minimal(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        ctrl = self._require_started()
        raw = ctrl.read_once(timeout_ms)
        return {
            "timestamp": tuple(raw.timestamp),
            "qpos": np.asarray(raw.q, dtype=float),
            "qvel": np.asarray(raw.dtheta, dtype=float),
            "dq": np.asarray(raw.dq, dtype=float),
            "theta": np.asarray(raw.theta, dtype=float),
            "dtheta": np.asarray(raw.dtheta, dtype=float),
            "tau": np.asarray(raw.tau, dtype=float),
            "tau_des": np.asarray(raw.tau_des, dtype=float),
            "tau_ext": np.asarray(raw.tau_ext, dtype=float),
            "last_torque": self._last_or_raw_torque(raw),
        }

    @property
    def state(self) -> dict[str, np.ndarray]:
        ctrl = self._require_started()
        if getattr(self, "model_backend", "rdk") == "mujoco":
            raw = ctrl.read_once(1000)
            return self._mujoco_state_from_raw(raw)

        raw = dict(ctrl.read_once_full(self.link_name, 1000))
        state = {
            key: np.asarray(value, dtype=float)
            for key, value in raw.items()
            if key != "timestamp"
        }
        state["timestamp"] = tuple(raw["timestamp"])
        state["last_torque"] = (
            self._last_torque.copy()
            if self._last_torque is not None
            else np.asarray(state["tau_des"], dtype=float)
        )
        return state

    def _mujoco_state_from_raw(self, raw) -> dict[str, np.ndarray]:
        backend = self._mujoco_backend
        if backend is None:
            raise RuntimeError("MuJoCo model backend has not been initialized")

        qpos = np.asarray(raw.q, dtype=float)
        dq = np.asarray(raw.dq, dtype=float)
        dtheta = np.asarray(raw.dtheta, dtype=float)
        qvel = dtheta if self.mujoco_velocity_source == "dtheta" else dq
        model_state = backend.state(qpos, qvel)
        state = {
            "timestamp": tuple(raw.timestamp),
            "qpos": qpos,
            "qvel": qvel.copy(),
            "dq": dq,
            "theta": np.asarray(raw.theta, dtype=float),
            "dtheta": dtheta,
            "tau": np.asarray(raw.tau, dtype=float),
            "tau_des": np.asarray(raw.tau_des, dtype=float),
            "tau_ext": np.asarray(raw.tau_ext, dtype=float),
            "tcp_pose": self._raw_array(raw, "tcp_pose", 7),
            "tcp_vel": self._raw_array(raw, "tcp_vel", 6),
            "flange_pose": self._raw_array(raw, "flange_pose", 7),
            "ft_sensor_raw": self._raw_array(raw, "ft_sensor_raw", 6),
            "ext_wrench_in_tcp": self._raw_array(raw, "ext_wrench_in_tcp", 6),
            "ext_wrench_in_world": self._raw_array(raw, "ext_wrench_in_world", 6),
            "ext_wrench_in_tcp_raw": self._raw_array(
                raw, "ext_wrench_in_tcp_raw", 6
            ),
            "ext_wrench_in_world_raw": self._raw_array(
                raw, "ext_wrench_in_world_raw", 6
            ),
            "last_torque": self._last_or_raw_torque(raw),
            "model_backend": np.asarray(["mujoco"], dtype=object),
            "mujoco_velocity_source": np.asarray(
                [self.mujoco_velocity_source], dtype=object
            ),
        }
        state.update(model_state)
        return state

    def step(self, torque: np.ndarray | list[float]) -> None:
        ctrl = self._require_started()
        tau = np.asarray(torque, dtype=float)
        if tau.shape != (self.dof,):
            raise ValueError(f"Expected torque shape {(self.dof,)}, got {tau.shape}")
        ctrl.write_once(
            tau.tolist(),
            self.enable_gravity_comp,
            self.enable_soft_limits,
            self.friction_comp_scale,
        )
        self._last_torque = tau.copy()
