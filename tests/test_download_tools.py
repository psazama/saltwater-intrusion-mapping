import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("geopandas")

import rasterio

from shapely.geometry import shape

from swmaps.core import download_tools


def _load_somerset_polygon() -> shape:
    somerset_path = Path("config/somerset.geojson")
    with somerset_path.open("r", encoding="utf-8") as fp:
        geojson = json.load(fp)
    return shape(geojson["features"][0]["geometry"])


@pytest.mark.integration
@pytest.mark.network
def test_download_nlcd_and_cdl_for_somerset(monkeypatch, tmp_path):
    polygon = _load_somerset_polygon()

    def tmp_data_path(*parts: str) -> Path:
        path = tmp_path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(download_tools, "data_path", tmp_data_path)

    nlcd_path = download_tools.download_nlcd(polygon, 2021, overwrite=True)
    assert nlcd_path.exists()
    assert nlcd_path.suffix.lower() == ".tif"
    assert nlcd_path.stat().st_size > 0

    with rasterio.open(nlcd_path) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.count >= 1
        assert dataset.width > 0
        assert dataset.height > 0

    cdl_path = download_tools.download_nass_cdl(polygon, 2020, overwrite=True)
    assert cdl_path.exists()
    assert cdl_path.suffix.lower() == ".tif"
    assert cdl_path.stat().st_size > 0

    with rasterio.open(cdl_path) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.count >= 1
        assert dataset.width > 0
        assert dataset.height > 0
