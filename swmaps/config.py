"""Configuration helpers for locating project data directories."""

from os import PathLike
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Parameters:
        None

    Attributes:
        data_root (Path): Root directory for persistent data artifacts.
    """

    # Can be overridden with SW_DATA_ROOT=/mnt/bucket or helm env var
    data_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    class Config:
        """Pydantic metadata configuring the ``SW_`` environment variable prefix.

        Parameters:
            None

        Attributes:
            env_prefix (str): Prefix applied to environment variables.
        """

        env_prefix = "SW_"


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance using current environment values.

    Args:
        None

    Returns:
        Settings: Settings object populated from environment variables.
    """

    return Settings()


settings = Settings()


def data_path(*parts: str | PathLike[str]) -> Path:
    """Convenience for building paths inside the :data:`data_root`.

    Args:
        *parts (str | os.PathLike): Path components joined relative to
            :attr:`Settings.data_root`.

    Returns:
        Path: Absolute path inside the configured data root.
    """

    return settings.data_root.joinpath(*parts)
