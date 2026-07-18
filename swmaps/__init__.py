"""Core utilities for mapping saltwater intrusion."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["config", "core"]

try:
    # Single source of truth: [project] version in pyproject.toml
    __version__ = version("swmaps")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0.dev0"
