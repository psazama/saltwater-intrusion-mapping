# swmaps/config.py
from os import PathLike
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Can be overridden with SW_DATA_ROOT=/mnt/bucket or helm env var
    data_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "data"
    )

    class Config:
        env_prefix = "SW_"


def get_settings():
    return Settings()


settings = Settings()


def data_path(*parts: str | PathLike[str]) -> Path:
    """Convenience for building paths inside the :data:`data_root`."""

    return settings.data_root.joinpath(*parts)
