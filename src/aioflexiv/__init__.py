from __future__ import annotations

from importlib.metadata import version

from ._loader import load_rt
from .controller import FlexivController
from .robot import FlexivRobotInterface

__version__ = version("aioflexiv")

__all__ = [
    "__version__",
    "FlexivController",
    "FlexivRobotInterface",
    "load_rt",
]
