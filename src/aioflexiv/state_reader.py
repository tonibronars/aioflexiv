"""Read-only Flexiv robot state, without enabling or commanding the robot.

FlexivStateReader opens its own RDK connection. It never enables the robot,
clears faults, switches mode or calls Stop, so it can observe a robot that
another process controls. Closing it only releases the connection.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .robot import measured_state_from_rdk


class FlexivStateReader:
    """Observe a Flexiv robot through the Python RDK without controlling it."""

    def __init__(
        self,
        robot_sn: str,
        *,
        network_interface_whitelist: Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        import flexivrdk

        # A full (non-lite) connection is needed to receive states.
        # No Enable, ClearFault, SwitchMode or Stop call follows.
        self._robot: Any | None = flexivrdk.Robot(
            robot_sn,
            list(network_interface_whitelist or []),
            bool(verbose),
            False,
        )

    def _require_open(self) -> Any:
        if self._robot is None:
            raise RuntimeError("The Flexiv state reader is closed")
        return self._robot

    @property
    def info(self) -> dict[str, Any]:
        """Copy robot metadata and the reported joint limits."""
        raw = self._require_open().info()
        return {
            "serial_num": str(raw.serial_num),
            "model_name": str(raw.model_name),
            "software_ver": str(raw.software_ver),
            "license_type": str(raw.license_type),
            "has_FT_sensor": bool(raw.has_FT_sensor),
            "dof": int(raw.DoF),
            "external_dof": int(raw.DoF_e),
            "manipulator_dof": int(raw.DoF_m),
            "q_min": [float(value) for value in raw.q_min],
            "q_max": [float(value) for value in raw.q_max],
            "dq_max": [float(value) for value in raw.dq_max],
            "tau_max": [float(value) for value in raw.tau_max],
            "K_q_nom": [float(value) for value in raw.K_q_nom],
            "K_x_nom": [float(value) for value in raw.K_x_nom],
        }

    def read(self) -> dict[str, Any]:
        """Copy the latest state, with the same keys as ``state_minimal()``.

        ``timestamp_monotonic`` is the local ``time.monotonic()`` receipt time.
        The RDK can return the same state twice, so callers that need fresh
        samples must check that ``timestamp`` advances.
        """
        robot = self._require_open()
        if not robot.connected():
            raise RuntimeError("The Flexiv state connection is disconnected")
        before = time.monotonic()
        raw = robot.states()
        after = time.monotonic()
        if tuple(raw.timestamp) == (0, 0):
            raise RuntimeError("Flexiv has not returned a timestamped robot state yet")
        state = measured_state_from_rdk(raw)
        state["timestamp_monotonic"] = (before + after) / 2
        return state

    def status(self) -> dict[str, Any]:
        """Read status flags without changing mode or clearing faults.

        The Python RDK 1.9 mode enum has no real-time modes, so a robot in
        RT_JOINT_TORQUE reports its mode as ``"???"``.
        """
        robot = self._require_open()
        return {
            "mode": str(robot.mode().name),
            "stopped": bool(robot.stopped()),
            "connected": bool(robot.connected()),
            "fault": bool(robot.fault()),
            "operational": bool(robot.operational()),
            "estop_released": bool(robot.estop_released()),
        }

    def close(self) -> None:
        self._robot = None

    def __enter__(self) -> FlexivStateReader:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
