# swmaps/config.py
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


settings = Settings()


def data_path(*parts) -> Path:
    """Convenience for building paths inside the data_root."""
    return settings.data_root.joinpath(*parts)
