from __future__ import annotations

from types import SimpleNamespace
import threading
import unittest

import numpy as np

from aioflexiv.controller import FlexivController


class OscTorqueTests(unittest.TestCase):
    def test_osc_damping_uses_controlled_link_jacobian_velocity(self) -> None:
        controller = FlexivController.__new__(FlexivController)
        controller.robot = SimpleNamespace(dof=7)
        controller.state_lock = threading.Lock()
        controller.ee_desired = np.eye(4)
        controller.ee_kp = np.zeros(6)
        controller.ee_kd = np.ones(6)
        controller.null_kp = np.zeros(7)
        controller.null_kd = np.zeros(7)
        controller.initial_qpos = np.zeros(7)

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

        np.testing.assert_allclose(tau[:6], -qvel[:6])
        self.assertEqual(tau[6], 0.0)


if __name__ == "__main__":
    unittest.main()
