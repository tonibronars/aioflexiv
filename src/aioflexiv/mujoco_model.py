from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _model_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / "models",
        Path.cwd() / "models",
        Path(sys.prefix) / "share" / "aioflexiv" / "models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_mujoco_model_path(model_name: str | None = None) -> Path:
    normalized = "" if model_name is None else model_name.lower().replace(" ", "")
    model_dir = _model_root()

    if "rizon4s" in normalized or "a02ls" in normalized:
        return model_dir / "flexiv_rizon4s" / "flexiv_rizon4s.xml"
    if "rizon4" in normalized or "a02l" in normalized:
        return model_dir / "flexiv_rizon4" / "flexiv_rizon4.xml"
    return model_dir / "flexiv_rizon4" / "flexiv_rizon4.xml"


def default_mujoco_scene_path(model_name: str | None = None) -> Path:
    model_path = default_mujoco_model_path(model_name)
    scene_path = model_path.with_name("scene.xml")
    return scene_path if scene_path.exists() else model_path


class MujocoModelBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        site_name: str | None = None,
        body_name: str = "link7",
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo model backend requires the `mujoco` Python package."
            ) from exc

        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo model XML not found: {path}")

        self._mujoco = mujoco
        self.path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._gravity_data = mujoco.MjData(self.model)

        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError(
                f"Expected a 7-DoF Flexiv model, got nq={self.model.nq}, "
                f"nv={self.model.nv} from {path}"
            )

        self.site_name = site_name
        self.body_name = body_name
        if site_name is not None:
            self.frame_type = "site"
            self.frame_name = site_name
            self.frame_id = self.model.site(site_name).id
        else:
            self.frame_type = "body"
            self.frame_name = body_name
            self.frame_id = self.model.body(body_name).id

    def state(self, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, np.ndarray]:
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos must have shape {(self.model.nq,)}, got {qpos.shape}"
            )
        if qvel.shape != (self.model.nv,):
            raise ValueError(
                f"qvel must have shape {(self.model.nv,)}, got {qvel.shape}"
            )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, self.data)

        ee = self._ee()
        jac = self._jacobian()
        mm = self._mass_matrix(self.data)
        bias = np.asarray(self.data.qfrc_bias, dtype=float).copy()
        gravity = self._gravity(qpos)

        return {
            "ee": ee,
            "jac": jac,
            "mm": mm,
            "coriolis": bias - gravity,
            "gravity": gravity,
            "mujoco_model_path": np.asarray([str(self.path)], dtype=object),
            "mujoco_frame_type": np.asarray([self.frame_type], dtype=object),
            "mujoco_frame_name": np.asarray([self.frame_name], dtype=object),
        }

    def gravity(self, qpos: np.ndarray) -> np.ndarray:
        return self._gravity(np.asarray(qpos, dtype=float))

    def _ee(self) -> np.ndarray:
        ee = np.eye(4)
        if self.frame_type == "site":
            frame = self.data.site(self.frame_id)
        else:
            frame = self.data.body(self.frame_id)
        ee[:3, :3] = frame.xmat.reshape(3, 3)
        ee[:3, 3] = frame.xpos
        return ee

    def _jacobian(self) -> np.ndarray:
        jac = np.zeros((6, self.model.nv))
        if self.frame_type == "site":
            self._mujoco.mj_jacSite(
                self.model, self.data, jac[:3], jac[3:], self.frame_id
            )
        else:
            self._mujoco.mj_jacBody(
                self.model, self.data, jac[:3], jac[3:], self.frame_id
            )
        return jac

    def _mass_matrix(self, data) -> np.ndarray:
        mm = np.zeros((self.model.nv, self.model.nv))
        self._mujoco.mj_fullM(self.model, mm, data.qM)
        return mm

    def _gravity(self, qpos: np.ndarray) -> np.ndarray:
        self._gravity_data.qpos[:] = qpos
        self._gravity_data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self._gravity_data)
        return np.asarray(self._gravity_data.qfrc_bias, dtype=float).copy()
