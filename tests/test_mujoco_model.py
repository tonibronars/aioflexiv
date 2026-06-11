from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from aioflexiv.mujoco_model import (
    MujocoModelBackend,
    default_mujoco_model_path,
    default_mujoco_scene_path,
)


@unittest.skipIf(importlib.util.find_spec("mujoco") is None, "mujoco is not installed")
class MujocoModelBackendTests(unittest.TestCase):
    def test_default_model_path_follows_robot_model_name(self) -> None:
        self.assertEqual(
            default_mujoco_model_path("Rizon4").name,
            "flexiv_rizon4.xml",
        )
        self.assertEqual(
            default_mujoco_model_path("Rizon4S").name,
            "flexiv_rizon4s.xml",
        )
        self.assertEqual(default_mujoco_scene_path("Rizon4").name, "scene.xml")

    def test_backend_computes_kinematics_and_dynamics_for_bundled_models(self) -> None:
        for model_name in ("Rizon4", "Rizon4S"):
            with self.subTest(model_name=model_name):
                backend = MujocoModelBackend(default_mujoco_model_path(model_name))
                state = backend.state(np.zeros(7), np.zeros(7))

                self.assertEqual(state["ee"].shape, (4, 4))
                self.assertEqual(state["jac"].shape, (6, 7))
                self.assertEqual(state["mm"].shape, (7, 7))
                self.assertEqual(state["coriolis"].shape, (7,))
                self.assertEqual(state["gravity"].shape, (7,))

                np.testing.assert_allclose(state["ee"][3], [0.0, 0.0, 0.0, 1.0])
                mass_eigs = np.linalg.eigvalsh(0.5 * (state["mm"] + state["mm"].T))
                self.assertGreater(float(mass_eigs[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
