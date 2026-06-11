from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from aioflexiv.robot import MujocoRobotInterface


@unittest.skipIf(importlib.util.find_spec("mujoco") is None, "mujoco is not installed")
class MujocoRobotInterfaceTests(unittest.TestCase):
    def test_start_state_and_step_use_mujoco_physics(self) -> None:
        robot = MujocoRobotInterface(mujoco_viewer=False, mujoco_realtime=False)
        robot.start()
        try:
            self.assertTrue(robot.started)
            self.assertEqual(robot.info["serial_num"], "mujoco")
            self.assertEqual(robot.dof, 7)
            self.assertEqual(robot.info["mujoco_timestep"], 0.001)
            self.assertEqual(robot.model.opt.timestep, 0.001)
            np.testing.assert_allclose(
                robot.torque_limit,
                [123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0],
            )

            state = robot.state
            self.assertEqual(state["model_backend"][0], "mujoco")
            self.assertEqual(state["ee"].shape, (4, 4))
            self.assertEqual(state["jac"].shape, (6, 7))
            self.assertEqual(state["mm"].shape, (7, 7))
            self.assertEqual(state["qpos"].shape, (7,))
            self.assertEqual(state["mujoco_frame_type"][0], "body")
            self.assertEqual(state["mujoco_frame_name"][0], "link7")
            self.assertEqual(state["mujoco_timestep"][0], 0.001)
            start_time = robot.data.time

            command = np.ones(robot.dof) * 0.1
            robot.step(command)

            self.assertGreater(robot.data.time, start_time)
            np.testing.assert_allclose(robot.state_minimal()["last_torque"], command)
        finally:
            robot.stop()


if __name__ == "__main__":
    unittest.main()
