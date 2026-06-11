from __future__ import annotations

from types import SimpleNamespace
import threading
import unittest

import numpy as np

from aioflexiv.controller import FlexivController


def _rz(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class OscTorqueTests(unittest.TestCase):
    def _controller(self) -> FlexivController:
        controller = FlexivController.__new__(FlexivController)
        controller.robot = SimpleNamespace(dof=7)
        controller.state_lock = threading.Lock()
        controller.ee_desired = np.eye(4)
        controller.ee_kp = np.zeros(6)
        controller.ee_kd = np.ones(6)
        controller.null_kp = np.zeros(7)
        controller.null_kd = np.zeros(7)
        controller.initial_qpos = np.zeros(7)
        controller.osc_pinv_damping = 1e-3
        controller.osc_pinv_rcond = 1e-6
        return controller

    def test_osc_damping_uses_controlled_link_jacobian_velocity(self) -> None:
        controller = self._controller()

        jac = np.zeros((6, 7))
        jac[:, :6] = np.eye(6)
        qvel = np.arange(1.0, 8.0)
        state = {
            "qpos": np.zeros(7),
            "qvel": qvel,
            "ee": np.eye(4),
            "jac": jac,
            "mm": np.eye(7),
            "tcp_vel": np.ones(6) * 100.0,
        }

        tau = controller._osc_torque(state)

        np.testing.assert_allclose(tau[:6], -qvel[:6], rtol=2e-6)
        self.assertEqual(tau[6], 0.0)

    def test_osc_task_inverse_is_damped_near_singular_orientation_axis(self) -> None:
        controller = self._controller()
        controller.ee_kp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 128.0])
        controller.ee_kd = np.zeros(6)

        ee = np.eye(4)
        ee[:3, :3] = _rz(0.1)
        jac = np.zeros((6, 7))
        jac[:5, :5] = np.eye(5)
        jac[5, 5] = 1e-3
        state = {
            "qpos": np.zeros(7),
            "qvel": np.zeros(7),
            "ee": ee,
            "jac": jac,
            "mm": np.eye(7),
        }

        tau = controller._osc_torque(state)

        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertLess(abs(tau[5]), 1.0)
        np.testing.assert_allclose(tau[5], 0.0)


if __name__ == "__main__":
    unittest.main()
