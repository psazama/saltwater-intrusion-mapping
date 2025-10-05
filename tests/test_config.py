"""Tests validating configuration helpers and environment overrides."""

import importlib.util
from pathlib import Path

import swmaps.config
from swmaps.config import get_settings

# Import swmaps.config without executing swmaps.__init__
CONFIG_PATH = Path(__file__).resolve().parents[1] / "swmaps" / "config.py"
spec = importlib.util.spec_from_file_location("swmaps.config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
def test_data_path_default() -> None:
    """Ensure data_path builds paths under the configured data root by default.

    Args:
        None

    Returns:
        None: Assertions verify the constructed paths.
    """
    root = config.settings.data_root
    assert config.data_path("subdir", "file.txt") == root / "subdir" / "file.txt"


def test_data_path_env_override(monkeypatch) -> None:
    """Verify environment variables override the default data root setting.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to set environment variables.

    Returns:
        None: Assertions ensure overridden paths are honoured and cleanup occurs.
    """
    tmp_root = Path("temp_data_root")
    monkeypatch.setenv("SW_DATA_ROOT", str(tmp_root))
    import importlib

    settings = get_settings()

    assert settings.data_root == tmp_root
    assert settings.data_root / "a" == tmp_root / "a"
    monkeypatch.delenv("SW_DATA_ROOT", raising=False)
    importlib.reload(swmaps.config)
