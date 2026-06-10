from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._loader import load_rt
from .controller import FlexivController
from .robot import FlexivRobotInterface

try:
    __version__ = version("aioflexiv")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "FlexivController",
    "FlexivRobotInterface",
    "load_rt",
]
