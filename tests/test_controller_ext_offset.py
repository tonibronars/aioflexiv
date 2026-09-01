from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from aioflexiv.controller import FlexivController
from aioflexiv.robot import FlexivRobotInterface


class _FakeFlexivRobot(FlexivRobotInterface):
    def __init__(self, idle_samples: list[np.ndarray]) -> None:
        self.events: list[str] = []
        self.idle_samples = [np.asarray(sample, dtype=float) for sample in idle_samples]
        self.idle_sample_index = 0
        self._started = False
        self._in_idle_mode = False
        self.written_torques: list[np.ndarray] = []
        self.post_switch_q = np.arange(7, dtype=float) * 0.01

    @property
    def started(self) -> bool:
        return self._started

    @property
    def dof(self) -> int:
        return 7

    @property
    def torque_limit(self) -> np.ndarray:
        return np.ones(7) * 100.0

    @property
    def in_idle_mode(self) -> bool:
        return self._in_idle_mode

    def start(self, *, defer_torque_mode: bool = False) -> None:
        self.events.append(f"start:{defer_torque_mode}")
        self._started = True
        self._in_idle_mode = defer_torque_mode

    def start_torque_control(self) -> None:
        self.events.append("start_torque_control")
        self._in_idle_mode = False

    def state_minimal(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        del timeout_ms
        if self._in_idle_mode:
            self.events.append("idle_state")
            index = min(self.idle_sample_index, len(self.idle_samples) - 1)
            tau_ext = self.idle_samples[index]
            self.idle_sample_index += 1
        else:
            self.events.append("torque_state")
            tau_ext = np.zeros(7)
        return {
            "qpos": self.post_switch_q.copy(),
            "qvel": np.zeros(7),
            "tau_ext": tau_ext.copy(),
        }

    @property
    def state(self) -> dict[str, np.ndarray]:
        self.events.append("full_torque_state")
        return {
            "qpos": self.post_switch_q.copy(),
            "qvel": np.zeros(7),
            "tau_ext": np.zeros(7),
            "ee": np.eye(4),
        }

    def step(self, torque: np.ndarray | list[float]) -> None:
        self.written_torques.append(np.asarray(torque, dtype=float))
        time.sleep(0.001)

    def stop(self) -> None:
        self.events.append("stop")
        self._started = False
        self._in_idle_mode = False


class ExternalTorqueOffsetTests(unittest.IsolatedAsyncioTestCase):
    def _controller(
        self,
        robot: _FakeFlexivRobot,
        *,
        duration_s: float = 0.0,
        ext_offset: bool = True,
    ) -> FlexivController:
        controller = FlexivController(
            "mujoco",
            mujoco_viewer=False,
            ext_offset=ext_offset,
            ext_offset_idle_average_s=duration_s,
        )
        controller.robot = robot
        return controller

    def test_default_idle_average_is_half_a_second(self) -> None:
        controller = FlexivController("mujoco", mujoco_viewer=False)

        self.assertEqual(controller.ext_offset_idle_average_s, 0.5)

    def test_idle_capture_averages_every_fresh_sample_in_window(self) -> None:
        samples = [
            np.ones(7),
            np.ones(7) * 3.0,
            np.ones(7) * 5.0,
        ]
        robot = _FakeFlexivRobot(samples)
        controller = self._controller(robot, duration_s=0.5)
        robot.start(defer_torque_mode=True)
        controller._sync_shapes()

        with patch(
            "aioflexiv.controller.time.perf_counter",
            side_effect=[0.0, 0.1, 0.2, 0.5, 0.6],
        ):
            controller._capture_idle_tau_ext_offset()

        np.testing.assert_allclose(controller.tau_ext_offset, np.ones(7) * 3.0)
        self.assertEqual(controller.tau_ext_offset_sample_count, 3)
        self.assertAlmostEqual(controller.tau_ext_offset_capture_duration_s, 0.6)

    async def test_start_captures_idle_mean_before_switching_torque_mode(self) -> None:
        idle_offset = np.arange(7, dtype=float) + 1.0
        robot = _FakeFlexivRobot([idle_offset])
        controller = self._controller(robot)

        await controller.start()
        try:
            self.assertEqual(
                robot.events[:4],
                [
                    "start:True",
                    "idle_state",
                    "start_torque_control",
                    "full_torque_state",
                ],
            )
            np.testing.assert_allclose(controller.tau_ext_offset, idle_offset)
            np.testing.assert_allclose(controller.q_desired, robot.post_switch_q)
            self.assertGreater(len(robot.written_torques), 0)
            np.testing.assert_allclose(robot.written_torques[0], -idle_offset)
        finally:
            await controller.stop()

    async def test_disabling_offset_uses_existing_direct_start(self) -> None:
        robot = _FakeFlexivRobot([np.ones(7)])
        controller = self._controller(robot, ext_offset=False)

        await controller.start()
        try:
            self.assertEqual(robot.events[0], "start:False")
            self.assertNotIn("idle_state", robot.events)
            np.testing.assert_allclose(controller.tau_ext_offset, np.zeros(7))
            self.assertEqual(controller.tau_ext_offset_sample_count, 0)
        finally:
            await controller.stop()

    def test_rejects_invalid_idle_average_duration(self) -> None:
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    FlexivController(
                        "mujoco",
                        mujoco_viewer=False,
                        ext_offset_idle_average_s=value,
                    )


if __name__ == "__main__":
    unittest.main()
