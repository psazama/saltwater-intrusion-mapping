"""Raster utility helpers for reprojection, resampling, and diagnostics."""

import logging
import os
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon


def warp_to_wgs84(
    src_path: str,
    dst_path: Optional[str] = None,
    resampling: Resampling = Resampling.nearest,
    dst_nodata: float = np.nan,
) -> str:
    """Reproject a raster to WGS‑84 (EPSG:4326) for web‑map visualisation.

    Parameters
    ----------
    src_path : str
        Path to the input GeoTIFF (any projected or geographic CRS).
    dst_path : str, optional
        Path for the re‑projected output. If *None* (default) the function
        appends ``_wgs84`` to *src_path* before the file extension.
    resampling : rasterio.enums.Resampling, optional
        Resampling algorithm (default: ``Resampling.nearest``).
    dst_nodata : float, optional
        Nodata value written to the output (default: ``numpy.nan``).

    Returns
    -------
    str
        Path to the generated WGS‑84 GeoTIFF.
    """

    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = f"{base}_wgs84{ext}"

    with rasterio.open(src_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": dst_nodata,
            }
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=dst_nodata,
                )

    return dst_path


def downsample_to_landsat(
    data: np.ndarray,
    transform: Affine,
    crs,
    target_resolution: float = 30,
    resampling: Resampling = Resampling.average,
) -> tuple:
    """Resample a raster array to Landsat-like 30 metre resolution.

    Args:
        data (np.ndarray): Source array to resample.
        transform (Affine): Affine transform describing the source grid.
        crs: Coordinate reference system associated with ``data``.
        target_resolution (float): Desired output pixel size in metres.
        resampling (Resampling): Resampling algorithm to use.

    Returns:
        tuple[np.ndarray, Affine]: The resampled array and its affine
        transform.
    """

    src_res = max(abs(transform.a), abs(transform.e))
    scale = target_resolution / src_res
    dst_height = max(1, int(np.ceil(data.shape[0] / scale)))
    dst_width = max(1, int(np.ceil(data.shape[1] / scale)))
    dst_transform = transform * Affine.scale(scale)

    dst = np.empty((dst_height, dst_width), dtype=data.dtype)

    reproject(
        source=data,
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=resampling,
        dst_nodata=np.nan,
    )

    return dst, dst_transform


def init_logger(log_path: str = "process_log.txt") -> None:
    """Configure a process-aware logger that writes to disk and stdout.

    Args:
        log_path (str): File path where log messages will be appended.

    Returns:
        None: Logging is configured as a side effect.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(process)d] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )


def find_non_nan_window(
    tif_path: str,
    bands: list[int] | None = None,
    window_size: int = 512,
    stride: int = 256,
    threshold_ratio: float = 0.5,
) -> tuple[np.ndarray | list[np.ndarray], dict, Window] | None:
    """Find a mostly valid window within a raster dataset.

    Args:
        tif_path (str): Path to the raster file to inspect.
        bands (list[int] | None): One-based band indices to read. When
            ``None``, only band 1 is evaluated.
        window_size (int): Edge length of the candidate window in pixels.
        stride (int): Step size used while scanning the raster.
        threshold_ratio (float): Minimum fraction of valid pixels required
            within a window.

    Returns:
        tuple[np.ndarray | list[np.ndarray], dict, Window] | None: Tuple
        containing the extracted data array(s), an updated raster profile,
        and the chosen window, or ``None`` if no valid window is found.
    """
    with rasterio.open(tif_path) as src:
        scale_reflectance = False
        if "landsat" in tif_path:
            scale_reflectance = True

        width, height = src.width, src.height
        band_list = bands if bands is not None else [1]

        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                window = Window(x, y, window_size, window_size)

                # Read and validate bands
                data_list = []
                valid = True

                for b in band_list:
                    data = src.read(b, window=window)
                    if scale_reflectance:
                        # Landsat Collection 2 Level-2 surface reflectance (SR) products store reflectance as integers and apply the scale factor = 0.0000275, offset = -0.2 as per USGS Landsat documentation
                        data = data.astype(np.float32) * 0.0000275 - 0.2
                    data_list.append(data)

                    valid_mask = ~np.isnan(data)
                    if (
                        np.sum(valid_mask) < threshold_ratio * data.size
                        or np.sum(np.isclose(data, 0))
                        > (1 - threshold_ratio) * data.size
                    ):
                        valid = False
                        break  # this window is not good

                if valid:
                    print(f"Found valid window at x={x}, y={y}")
                    transform = src.window_transform(window)
                    profile = src.profile.copy()
                    profile.update(
                        {
                            "height": window.height,
                            "width": window.width,
                            "transform": transform,
                        }
                    )
                    profile.update(
                        {
                            "dtype": "float32",
                            "nodata": np.nan,
                        }
                    )

                    # Return single array if only one band, else list
                    return (
                        (data_list[0] if len(data_list) == 1 else data_list),
                        profile,
                        window,
                    )

    print("No valid window found.")
    return None, None, None


def reproject_bbox(
    bbox: (
        list[float]
        | tuple[float, float, float, float]
        | gpd.GeoDataFrame
        | gpd.GeoSeries
        | Polygon
        | MultiPolygon
    ),
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:32618",
) -> list[float]:
    """Project a bounding box or geometry from one CRS into another.

    Args:
        bbox (sequence | GeoDataFrame | GeoSeries | Polygon | MultiPolygon):
            Input bounds or geometry expressed in ``src_crs`` coordinates.
        src_crs (str): Coordinate reference system describing ``bbox``.
        dst_crs (str): Target coordinate reference system for the output bounds.

    Returns:
        list[float]: Reprojected bounding box ``[minx, miny, maxx, maxy]`` in
        ``dst_crs``.
    """

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        geom = None
        minx, miny = bbox[0], bbox[1]
        maxx, maxy = bbox[2], bbox[3]
    else:
        # shapely geometry OR Geo(Data)Frame
        if hasattr(bbox, "geometry"):  # GeoDataFrame / GeoSeries
            geom = bbox.unary_union
        else:  # Polygon / MultiPolygon
            geom = bbox
        minx, miny, maxx, maxy = geom.bounds

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    minx_proj, miny_proj = transformer.transform(minx, miny)
    maxx_proj, maxy_proj = transformer.transform(maxx, maxy)

    return [
        min(minx_proj, maxx_proj),
        min(miny_proj, maxy_proj),
        max(minx_proj, maxx_proj),
        max(miny_proj, maxy_proj),
    ]
