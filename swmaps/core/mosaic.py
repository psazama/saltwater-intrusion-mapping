"""GEE-based raster mosaicking utilities.

Replaces the old STAC-based per-band downloading and stacking logic with
Earth Engine imagery retrieval and export. Outputs are local multiband
GeoTIFFs so the rest of the pipeline (salinity, NDWI, model features)
continues to work unchanged.

Public functions
----------------
:func:`compute_bbox` - build a bounding box around a lat/lon point.
:func:`process_date` - download a single mosaic for a location and date.
:func:`process_multiple` - apply :func:`process_date` to every row of a DataFrame.
"""

from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.missions import get_mission
from swmaps.core.satellite_query import (
    download_gee_multiband,
    get_best_image,
    query_gee_images,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def compute_bbox(lat, lon, buffer_km=1.0):
    """Return a bounding box around a centre point with a buffer distance.

    Args:
        lat: Centre latitude in decimal degrees.
        lon: Centre longitude in decimal degrees.
        buffer_km: Half-width of the bounding box in kilometres.
            Defaults to ``1.0``.

    Returns:
        list[float]: ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    """
    deg = buffer_km / 111.0
    return [lon - deg, lat - deg, lon + deg, lat + deg]


def _write_rgb_png_from_tif(
    tif_path: Path,
    png_path: Path,
    rgb_bands: tuple[int, int, int],
    stretch: bool = True,
):
    """Save an RGB PNG preview from a multiband GeoTIFF.

    Args:
        tif_path: Path to the source multiband GeoTIFF.
        png_path: Destination path for the output PNG.
        rgb_bands: 1-based band indices for red, green, blue channels
            respectively, e.g. ``(4, 3, 2)`` for Sentinel-2.
        stretch: When ``True``, applies a 2–98 percentile min-max stretch
            for improved visual contrast. Defaults to ``True``.

    Returns:
        None
    """
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(tif_path) as src:
        r = src.read(rgb_bands[0]).astype("float32")
        g = src.read(rgb_bands[1]).astype("float32")
        b = src.read(rgb_bands[2]).astype("float32")

    def _norm(x):
        if not stretch:
            return x
        lo, hi = np.percentile(x, (2, 98))
        if hi <= lo:
            return np.zeros_like(x)
        x = (x - lo) / (hi - lo)
        return np.clip(x, 0, 1)

    r = _norm(r)
    g = _norm(g)
    b = _norm(b)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype("uint8")

    Image.fromarray(rgb, mode="RGB").save(png_path)


# ---------------------------------------------------------------------
# Main GEE Mosaic Build
# ---------------------------------------------------------------------


def process_date(
    lat: float,
    lon: float,
    date: datetime,
    buffer_km: float = 1.0,
    mission: str = "sentinel-2",
    out_dir: str | Path | None = None,
    days_before: int = 7,
    days_after: int = 7,
    cloud_filter: float = 30,
    samples: int = 1,
    save_png: bool = False,
):
    """Download a multiband GeoTIFF mosaic for a location and date.

    Queries GEE for imagery within the temporal window, selects the
    best available scene(s), and exports a clipped multiband GeoTIFF.
    Optionally writes an RGB PNG preview alongside the raster.

    Args:
        lat: Centre latitude in decimal degrees.
        lon: Centre longitude in decimal degrees.
        date: Target acquisition date.
        buffer_km: Bounding-box half-width in kilometres. Defaults to ``1.0``.
        mission: Mission slug - ``"sentinel-2"``, ``"landsat-5"``, or
            ``"landsat-7"``. Defaults to ``"sentinel-2"``.
        out_dir: Directory where output files will be written. Defaults to
            ``<data_root>/mosaics``.
        days_before: Days before *date* to include in the search window.
            Defaults to ``7``.
        days_after: Days after *date* to include in the search window.
            Defaults to ``7``.
        cloud_filter: Maximum allowed cloud cover percentage. Defaults to
            ``30``.
        samples: Maximum number of scenes to return per date. Defaults to
            ``1``.
        save_png: Whether to write an RGB PNG preview. Defaults to
            ``False``.

    Returns:
        list[str] | None: Paths to the downloaded GeoTIFF(s), or ``None``
        if no suitable imagery was found.
    """
    if out_dir is None:
        out_dir = data_path("mosaics")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build date window
    dt = date
    date_range = (
        f"{(dt - timedelta(days=days_before)).date()}/"
        f"{(dt + timedelta(days=days_after)).date()}"
    )

    bbox = compute_bbox(lat, lon, buffer_km)
    mission_info = get_mission(mission)

    # Default RGB band choices per mission, 1-based indices
    if mission == "sentinel-2":
        rgb_bands = (4, 3, 2)  # B4 red, B3 green, B2 blue
    elif mission in ("landsat-5", "landsat-7"):
        rgb_bands = (3, 2, 1)  # red, green, blue
    else:
        rgb_bands = None

    # Query ImageCollection
    col, band_map = query_gee_images(
        mission=mission,
        bbox=bbox,
        date_range=date_range,
        cloud_filter=cloud_filter,
    )

    size = col.size().getInfo()
    if size == 0:
        return None

    # Select best image
    imgs = get_best_image(col, mission, samples)
    if imgs is None:
        print(
            f"No usable image found after filtering for {mission} around {date_range}"
        )
        return None

    output_paths = []
    print(f"Begin processing mosaic count: {len(imgs)}")
    for img in imgs:
        # Export clipped multiband raster
        output_path = download_gee_multiband(
            image=img,
            mission=mission,
            bands=band_map,
            bbox=bbox,
            out_dir=out_dir,
            scale=mission_info.gee_scale,
        )

        output_paths.append(output_path)
        print(f"[GEE] Wrote mosaic to: {output_path}")

        if save_png and rgb_bands is not None:
            png_path = Path(output_path).with_suffix(".png")
            _write_rgb_png_from_tif(
                tif_path=Path(output_path),
                png_path=png_path,
                rgb_bands=rgb_bands,
            )
            print(f"[GEE] Wrote RGB preview to: {png_path}")
    return output_paths


# ---------------------------------------------------------------------
# Batch processing for many dates/locations (optional API)
# ---------------------------------------------------------------------


def process_multiple(
    df,
    lat_col="latitude",
    lon_col="longitude",
    date_col="date",
    mission="sentinel-2",
    buffer_km=1.0,
    out_dir=None,
    days_before=7,
    days_after=7,
    cloud_filter=30,
    samples=1,
    save_png=False,
):
    """Apply :func:`process_date` to every row of a DataFrame.

    Args:
        df: DataFrame with at minimum *lat_col*, *lon_col*, and *date_col*
            columns.
        lat_col: Name of the latitude column. Defaults to ``"latitude"``.
        lon_col: Name of the longitude column. Defaults to ``"longitude"``.
        date_col: Name of the date column. Defaults to ``"date"``.
        mission: Mission slug. Defaults to ``"sentinel-2"``.
        buffer_km: Bounding-box half-width in kilometres. Defaults to
            ``1.0``.
        out_dir: Output directory for downloaded mosaics.
        days_before: Temporal window before each date in days. Defaults to
            ``7``.
        days_after: Temporal window after each date in days. Defaults to
            ``7``.
        cloud_filter: Maximum allowed cloud cover percentage. Defaults to
            ``30``.
        samples: Maximum scenes per date. Defaults to ``1``.
        save_png: Whether to write RGB PNG previews. Defaults to ``False``.

    Returns:
        list: List of output paths (one per row), with ``None`` entries
        where no imagery was found.
    """
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat = row[lat_col]
        lon = row[lon_col]
        date = row[date_col]
        path = process_date(
            lat=lat,
            lon=lon,
            date=date,
            mission=mission,
            buffer_km=buffer_km,
            out_dir=out_dir,
            days_before=days_before,
            days_after=days_after,
            cloud_filter=cloud_filter,
            samples=samples,
            save_png=save_png,
        )
        results.append(path)
    return results
