import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")

# Import swmaps.config without executing swmaps.__init__
CONFIG_PATH = Path(__file__).resolve().parents[1] / "swmaps" / "config.py"
spec = importlib.util.spec_from_file_location("swmaps.config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def test_data_path_default():
    root = config.settings.data_root
    assert config.data_path("subdir", "file.txt") == root / "subdir" / "file.txt"


def test_data_path_env_override(monkeypatch):
    tmp_root = Path("temp_data_root")
    monkeypatch.setenv("SW_DATA_ROOT", str(tmp_root))
    import importlib

    importlib.reload(config)
    assert config.settings.data_root == tmp_root
    assert config.data_path("a") == tmp_root / "a"
    monkeypatch.delenv("SW_DATA_ROOT", raising=False)
    importlib.reload(config)
