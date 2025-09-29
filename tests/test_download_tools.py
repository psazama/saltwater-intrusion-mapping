import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("geopandas")

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from shapely.geometry import shape

from swmaps.core import download_tools


def _load_somerset_polygon() -> shape:
    somerset_path = Path("config/somerset.geojson")
    with somerset_path.open("r", encoding="utf-8") as fp:
        geojson = json.load(fp)
    return shape(geojson["features"][0]["geometry"])


@pytest.fixture
def somerset_polygon() -> shape:
    return _load_somerset_polygon()


@pytest.fixture
def patched_data_path(monkeypatch, tmp_path):
    def tmp_data_path(*parts: str) -> Path:
        path = tmp_path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(download_tools, "data_path", tmp_data_path)
    return tmp_data_path


@pytest.mark.integration
@pytest.mark.network
def test_download_nlcd_for_somerset(monkeypatch, patched_data_path, somerset_polygon):
    bounds = somerset_polygon.bounds
    geotiff_bytes = _create_geotiff_bytes(bounds, width=16, height=16)
    sciencebase_calls: list[dict[str, object]] = []
    download_calls: list[dict[str, object]] = []

    class MockResponse:
        def __init__(self, *, content=b"", headers=None, json_data=None, status_code=200):
            self.content = content
            self.headers = headers or {}
            self._json_data = json_data
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            if self._json_data is None:
                raise ValueError("No JSON payload configured for MockResponse")
            return self._json_data

        def iter_content(self, chunk_size=8192):
            data = self.content or b""
            for start in range(0, len(data), chunk_size):
                yield data[start : start + chunk_size]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    expected_asset_url = "https://example.com/nlcd_2021_l48.tif"

    def fake_get(url, *args, **kwargs):
        if url == download_tools.ANNUAL_NLCD_SCIENCEBASE_ITEM:
            sciencebase_calls.append({"url": url, "kwargs": kwargs})
            return MockResponse(
                headers={"Content-Type": "application/json"},
                json_data={
                    "files": [
                        {
                            "name": "NLCD_2021_Land_Cover_L48.tif",
                            "url": expected_asset_url,
                        }
                    ]
                },
            )

        if url == expected_asset_url:
            download_calls.append({"url": url, "kwargs": kwargs})
            assert kwargs.get("stream") is True
            return MockResponse(
                content=geotiff_bytes,
                headers={"Content-Type": "image/tiff"},
            )

        raise AssertionError(f"Unexpected GET request for {url!r}")

    monkeypatch.setattr(download_tools.requests, "get", fake_get)

    nlcd_path = download_tools.download_nlcd(somerset_polygon, 2021, overwrite=True)
    assert nlcd_path.exists()
    assert nlcd_path.suffix.lower() == ".tif"
    assert nlcd_path.stat().st_size > 0

    with rasterio.open(nlcd_path) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.count >= 1
        assert dataset.width > 0
        assert dataset.height > 0

    assert sciencebase_calls, "Expected ScienceBase metadata request"
    assert download_calls, "Expected NLCD asset download request"
    assert download_calls[0]["kwargs"].get("timeout") == 300


@pytest.mark.integration
@pytest.mark.network
def test_download_cdl_for_somerset(monkeypatch, patched_data_path, somerset_polygon):
    bounds = somerset_polygon.bounds
    geotiff_bytes = _create_geotiff_bytes(bounds, width=20, height=12, value=2)
    captured: dict[str, object] = {}

    class MockResponse:
        def __init__(self, *, content=b"", headers=None, status_code=200):
            self.content = content
            self.headers = headers or {}
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

    def fake_get(url, *args, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return MockResponse(content=geotiff_bytes, headers={"Content-Type": "image/tiff"})

    monkeypatch.setattr(download_tools.requests, "get", fake_get)

    cdl_path = download_tools.download_nass_cdl(somerset_polygon, 2020, overwrite=True)
    assert cdl_path.exists()
    assert cdl_path.suffix.lower() == ".tif"
    assert cdl_path.stat().st_size > 0

    with rasterio.open(cdl_path) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.count >= 1
        assert dataset.width > 0
        assert dataset.height > 0

    assert captured["url"] == (
        "https://nassgeodata.gmu.edu/arcgis/rest/services/CDL/AnnualCDL/MapServer/export"
    )
    params = captured["kwargs"].get("params")
    assert params is not None
    assert params["bbox"] == ",".join(map(str, bounds))
    assert params["time"]
    assert captured["kwargs"].get("timeout") == 300


def _create_geotiff_bytes(
    bounds: tuple[float, float, float, float], *, width: int, height: int, value: int = 1
) -> bytes:
    transform = from_bounds(*bounds, width=width, height=height)
    data = np.full((height, width), value, dtype=np.uint8)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dataset:
            dataset.write(data, 1)
        return memfile.read()
