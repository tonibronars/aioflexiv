from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from aioflexiv.robot import FlexivRobotInterface


class RobotStateTests(unittest.TestCase):
    def test_state_exposes_force_torque_fields_as_arrays(self) -> None:
        raw_state = {
            "timestamp": (1, 2),
            "qpos": [0.0] * 7,
            "qvel": [0.0] * 7,
            "dq": [0.0] * 7,
            "theta": [0.0] * 7,
            "tau": [0.0] * 7,
            "tau_des": [0.0] * 7,
            "tau_ext": [0.0] * 7,
            "tcp_pose": [0.0] * 7,
            "tcp_vel": [0.0] * 6,
            "flange_pose": [0.0] * 7,
            "ft_sensor_raw": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ext_wrench_in_tcp": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "ext_wrench_in_world": [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            "ext_wrench_in_tcp_raw": [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "ext_wrench_in_world_raw": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            "ee": np.eye(4),
            "jac": np.zeros((6, 7)),
            "mm": np.eye(7),
            "coriolis": [0.0] * 7,
            "gravity": [0.0] * 7,
        }
        ctrl = SimpleNamespace(read_once_full=lambda link_name, timeout_ms: raw_state)

        robot = FlexivRobotInterface.__new__(FlexivRobotInterface)
        robot._ctrl = ctrl
        robot._last_torque = None
        robot.link_name = "flange"

        state = robot.state

        for key in (
            "ft_sensor_raw",
            "ext_wrench_in_tcp",
            "ext_wrench_in_world",
            "ext_wrench_in_tcp_raw",
            "ext_wrench_in_world_raw",
        ):
            self.assertIsInstance(state[key], np.ndarray)
            self.assertEqual(state[key].shape, (6,))

        np.testing.assert_allclose(state["ft_sensor_raw"], raw_state["ft_sensor_raw"])
        np.testing.assert_allclose(
            state["ext_wrench_in_tcp"], raw_state["ext_wrench_in_tcp"]
        )
        np.testing.assert_allclose(
            state["ext_wrench_in_world_raw"], raw_state["ext_wrench_in_world_raw"]
        )


if __name__ == "__main__":
    unittest.main()
