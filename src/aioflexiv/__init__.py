from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._loader import load_rt
from .controller import FlexivController
from .mujoco_model import MujocoModelBackend, default_mujoco_model_path
from .robot import FlexivRobotInterface

try:
    __version__ = version("aioflexiv")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "FlexivController",
    "FlexivRobotInterface",
    "MujocoModelBackend",
    "default_mujoco_model_path",
    "load_rt",
]
