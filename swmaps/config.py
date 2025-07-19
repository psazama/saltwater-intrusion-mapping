# swmaps/config.py
from os import PathLike
from pathlib import Path

from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover - optional dependency
    BaseSettings = None  # type: ignore[assignment]


if BaseSettings is not None:
    class Settings(BaseSettings):
        # Can be overridden with SW_DATA_ROOT=/mnt/bucket or helm env var
        data_root: Path = Field(
            default_factory=lambda: Path(__file__).resolve().parent / "data"
        )

        class Config:
            env_prefix = "SW_"
else:  # pragma: no cover - optional dependency missing
    Settings = None  # type: ignore[assignment]


def get_settings():
    if Settings is None:
        raise RuntimeError("pydantic_settings is required for get_settings")
    return Settings()


if Settings is not None:
    settings = Settings()
else:  # pragma: no cover - optional dependency missing
    settings = None


def data_path(*parts: str | PathLike[str]) -> Path:
    """Convenience for building paths inside the :data:`data_root`."""

    if settings is None:
        raise RuntimeError("pydantic_settings is required for data_path")
    return settings.data_root.joinpath(*parts)
