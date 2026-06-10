from __future__ import annotations

from importlib.metadata import version

from ._loader import load_rt

__version__ = version("aioflexiv")

__all__ = ["__version__", "load_rt"]

