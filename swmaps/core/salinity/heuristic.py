"""
Heuristic baseline for salinity classification from multispectral imagery.

Implements an index-based approach combining NDWI, MNDWI, turbidity,
chlorophyll, and SWIR proxies. Returns both continuous scores and
categorical class maps.
"""

import numpy as np

from swmaps.core.salinity.utils import _safe_normalized_difference


def estimate_salinity_level(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    *,
    reflectance_scale: float | None = 10000.0,
    water_threshold: float = 0.2,
    salinity_proxy_scale: float = 1.2,
    salinity_proxy_threshold: float = 0.35,
    chlorophyll_reference: float = 2.0,
) -> dict[str, np.ndarray]:
    """Estimate water salinity categories from multispectral bands.

    The heuristic combines the remote sensing proxies commonly cited for
    differentiating freshwater from saline or hypersaline water bodies:

    - Water detection from NDWI/MNDWI (green vs. NIR/SWIR)
    - Turbidity and Normalised Difference Turbidity Index (red vs. green/blue)
    - Chlorophyll proxies (green vs. blue)
    - Salinity proxy index from short-wave infrared reflectance (SWIR1/2)
    - Vegetation stress indicator via NDVI around the water pixel

    Inputs are expected to be surface reflectance bands from Sentinel-2, Landsat,
    or similar sensors. When the data are scaled (e.g., Sentinel-2 L2A stored as
    integers 0–10,000), ``reflectance_scale`` rescales the input into the
    0–1 range before the indices are computed.

    Parameters
    ----------
    blue, green, red, nir, swir1, swir2:
        Arrays representing the corresponding spectral bands. All arrays must
        share the same shape.
    reflectance_scale:
        If provided, each band is divided by this value to convert to
        reflectance. Set to ``None`` to skip rescaling.
    water_threshold:
        Threshold applied to NDWI/MNDWI to declare a pixel water-covered.
    salinity_proxy_scale:
        Normalising constant for the SWIR salinity proxy ``swir1 + swir2``.
    salinity_proxy_threshold:
        Pixels with a normalised salinity proxy above this value are also
        considered water (useful for bright saline pans with low NDWI).
    chlorophyll_reference:
        Reference ratio for the chlorophyll proxy. Values above this reference
        are treated as healthy (low salinity), whereas lower values indicate a
        potential salinity signal.

    Returns
    -------
    dict
        ``{"score", "class_map", "water_mask", "indices"}`` where

        - ``score`` is a float32 array (0–1) salinity intensity estimate with
          NaNs where water is not detected.
        - ``class_map`` is a string array with labels ``{"land", "fresh",
          "brackish", "saline"}``.
        - ``water_mask`` is a boolean array marking detected water pixels.
        - ``indices`` is a dictionary of the intermediate proxies used in the
          computation for transparency/debugging.
    """

    bands = [
        np.asarray(arr, dtype=np.float32, order="C")
        for arr in (blue, green, red, nir, swir1, swir2)
    ]

    if reflectance_scale:
        bands = [band / reflectance_scale for band in bands]

    blue_r, green_r, red_r, nir_r, swir1_r, swir2_r = bands

    ndwi = _safe_normalized_difference(green_r, nir_r)
    mndwi = _safe_normalized_difference(green_r, swir1_r)
    ndvi = _safe_normalized_difference(nir_r, red_r)
    ndti = _safe_normalized_difference(green_r, blue_r)

    turbidity_ratio = np.divide(
        red_r,
        green_r,
        out=np.zeros_like(red_r, dtype=np.float32),
        where=green_r != 0,
    ).astype(np.float32)
    chlorophyll_ratio = np.divide(
        green_r,
        blue_r,
        out=np.zeros_like(green_r, dtype=np.float32),
        where=blue_r != 0,
    ).astype(np.float32)

    salinity_proxy = np.clip(swir1_r + swir2_r, a_min=0.0, a_max=None).astype(
        np.float32
    )
    salinity_proxy_norm = np.clip(
        np.divide(
            salinity_proxy,
            salinity_proxy_scale,
            out=np.zeros_like(salinity_proxy, dtype=np.float32),
            where=salinity_proxy_scale != 0,
        ),
        0.0,
        1.0,
    )

    chlorophyll_norm = np.clip(
        np.divide(
            chlorophyll_ratio,
            chlorophyll_reference,
            out=np.zeros_like(chlorophyll_ratio, dtype=np.float32),
            where=chlorophyll_reference != 0,
        ),
        0.0,
        1.0,
    )
    turbidity_norm = np.clip(turbidity_ratio / 2.0, 0.0, 1.0)

    ndwi_scaled = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
    mndwi_scaled = np.clip((mndwi + 1.0) / 2.0, 0.0, 1.0)

    # Base water detection
    water_mask = (ndwi > water_threshold) | (mndwi > water_threshold)

    # Define weak water evidence
    weak_water = (ndwi > 0.0) | (mndwi > 0.0)

    # Allow salinity proxy to expand mask only where weak water evidence exists
    salinity_override = (salinity_proxy_norm > salinity_proxy_threshold) & weak_water
    water_mask = water_mask | salinity_override

    dryness_component = 1.0 - ndwi_scaled
    saline_surface_component = 1.0 - mndwi_scaled
    chlorophyll_deficit = 1.0 - chlorophyll_norm

    score = (
        0.2 * dryness_component
        + 0.15 * saline_surface_component
        + 0.45 * salinity_proxy_norm
        + 0.1 * turbidity_norm
        + 0.1 * chlorophyll_deficit
    )
    score = np.clip(score, 0.0, 1.0).astype(np.float32)

    score = np.where(water_mask, score, np.nan)
    score = score.astype(np.float32, copy=False)

    class_map = np.full(score.shape, "land", dtype="<U8")
    score_filled = np.nan_to_num(score, nan=0.0)
    class_map = np.where(
        water_mask,
        np.select(
            [score_filled < 0.35, score_filled < 0.6],
            ["fresh", "brackish"],
            default="saline",
        ),
        "land",
    )

    indices = {
        "ndwi": ndwi,
        "mndwi": mndwi,
        "ndvi": ndvi,
        "ndti": ndti,
        "turbidity_ratio": turbidity_ratio,
        "chlorophyll_ratio": chlorophyll_ratio,
        "salinity_proxy": salinity_proxy,
        "salinity_proxy_norm": salinity_proxy_norm,
    }

    return {
        "score": score,
        "class_map": class_map,
        "water_mask": water_mask,
        "indices": indices,
    }
