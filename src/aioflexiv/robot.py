from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from ._loader import load_rt
from .config import resolve_robot_sn, save_last_robot_sn
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

        self._rt = None
        self._ctrl = None
        self._info: dict[str, Any] | None = None
        self._last_torque: np.ndarray | None = None

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

    def _require_started(self):
        if self._ctrl is None:
            raise RuntimeError("Robot interface has not been started")
        return self._ctrl

    def state_minimal(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        ctrl = self._require_started()
        raw = ctrl.read_once(timeout_ms)
        last_torque = (
            self._last_torque.copy()
            if self._last_torque is not None
            else np.asarray(raw.tau_des, dtype=float)
        )
        return {
            "timestamp": tuple(raw.timestamp),
            "qpos": np.asarray(raw.q, dtype=float),
            "qvel": np.asarray(raw.dtheta, dtype=float),
            "dq": np.asarray(raw.dq, dtype=float),
            "theta": np.asarray(raw.theta, dtype=float),
            "tau": np.asarray(raw.tau, dtype=float),
            "tau_des": np.asarray(raw.tau_des, dtype=float),
            "tau_ext": np.asarray(raw.tau_ext, dtype=float),
            "last_torque": last_torque,
        }

    @property
    def state(self) -> dict[str, np.ndarray]:
        ctrl = self._require_started()
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
