from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.errors import RasterioError

from swmaps.models.base import BaseSalinityModel

# Map salinity classes to integer codes for compact raster storage
SALINITY_CLASS_CODES = {"land": 0, "fresh": 1, "brackish": 2, "saline": 3}


class SalinityHeuristicModel(BaseSalinityModel):
    """
    Heuristic (rule-based) salinity estimator implemented as a model.

    This class consolidates the heuristic computations and raster I/O helpers
    so you can run salinity estimation from in-memory band arrays or from
    on-disk mosaics (rasterio DatasetReaders / paths).

    The core scoring logic is a near-direct adaptation of the index-based
    heuristic used in swmaps.core.salinity.heuristic. This model is not
    trainable — it extends BaseSalinityModel for consistent typing with other
    models in the codebase.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__(output_dim=output_dim)

    # --------- Internal helpers (adapted) ----------
    @staticmethod
    def _safe_normalized_difference(
        numerator: np.ndarray, denominator: np.ndarray
    ) -> np.ndarray:
        """Compute a normalized difference index with zero-division protection."""
        return np.divide(
            numerator - denominator,
            numerator + denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=(numerator + denominator) != 0,
        ).astype(np.float32)

    @staticmethod
    def landsat_reflectance_stack(
        src: rasterio.io.DatasetReader,
    ) -> Dict[str, np.ndarray]:
        """
        Convert Landsat Collection 2 Level-2 DN values to surface reflectance bands.

        The ordering assumed is the Collection 2 convention:
          1 = coastal/aerosol
          2 = blue
          3 = green
          4 = red
          5 = nir
          6 = swir1
          7 = swir2

        Returns a dict with keys 'blue','green','red','nir','swir1','swir2' as float32 arrays.
        """
        scale = 0.0000275
        offset = -0.2

        # Defensive: ensure there are at least six bands
        if src.count < 6:
            raise ValueError("Expected >=6 bands for Landsat reflectance conversion")

        return {
            "blue": (src.read(2).astype(np.float32) * scale + offset),
            "green": (src.read(3).astype(np.float32) * scale + offset),
            "red": (src.read(4).astype(np.float32) * scale + offset),
            "nir": (src.read(5).astype(np.float32) * scale + offset),
            "swir1": (src.read(6).astype(np.float32) * scale + offset),
            "swir2": (src.read(7).astype(np.float32) * scale + offset),
        }

    @staticmethod
    def write_single_band(
        path: Path,
        profile: dict,
        array: np.ndarray,
        dtype: str,
        nodata: Optional[float | int] = np.nan,
    ) -> None:
        """Save a single-band raster to disk using a template profile."""
        profile = profile.copy()
        profile.update({"count": 1, "dtype": dtype})

        if nodata is None:
            profile.pop("nodata", None)
        else:
            profile["nodata"] = nodata

        path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(path, "w", **profile, BIGTIFF="YES") as dst:
            dst.write(array.astype(dtype, copy=False), 1)

    # --------- Prediction API ----------
    def predict_from_bands(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir1: np.ndarray,
        swir2: np.ndarray,
        *,
        reflectance_scale: Optional[float] = 10000.0,
        water_threshold: float = 0.2,
        salinity_proxy_scale: float = 1.2,
        salinity_proxy_threshold: float = 0.35,
        chlorophyll_reference: float = 2.0,
    ) -> Dict[str, np.ndarray]:
        """
        Run the heuristic salinity estimation on provided band arrays.

        Returns the same structure as the original heuristic:
            {'score', 'class_map', 'water_mask', 'indices'}
        """

        bands = [
            np.asarray(arr, dtype=np.float32, order="C")
            for arr in (blue, green, red, nir, swir1, swir2)
        ]

        if reflectance_scale:
            bands = [band / reflectance_scale for band in bands]

        blue_r, green_r, red_r, nir_r, swir1_r, swir2_r = bands

        ndwi = self._safe_normalized_difference(green_r, nir_r)
        mndwi = self._safe_normalized_difference(green_r, swir1_r)
        ndvi = self._safe_normalized_difference(nir_r, red_r)
        ndti = self._safe_normalized_difference(green_r, blue_r)

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
        salinity_override = (
            salinity_proxy_norm > salinity_proxy_threshold
        ) & weak_water
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

    def predict_from_rasterio(
        self,
        src: rasterio.io.DatasetReader,
        *,
        reflectance_scale: Optional[float] = None,
        water_threshold: float = 0.2,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: read bands from an open rasterio dataset and predict.

        By default this expects a Landsat Collection 2 Level-2 ordering. If you
        want to pass pre-extracted arrays, use predict_from_bands instead.
        """
        # If reflectance_scale is None, assume the dataset is already reflectance
        # (values in 0..1). If None, do not apply landsat conversion.
        if reflectance_scale is None:
            # Attempt to read commonly-named band indices if available (fall back to band numbers)
            try:
                blue = src.read(1).astype(np.float32)
                green = src.read(2).astype(np.float32)
                red = src.read(3).astype(np.float32)
                nir = src.read(4).astype(np.float32)
                swir1 = src.read(5).astype(np.float32)
                swir2 = src.read(6).astype(np.float32)
            except Exception as e:
                raise ValueError("Unable to read expected bands from dataset") from e
        else:
            # Landsat Collection 2 conversion
            bands = self.landsat_reflectance_stack(src)
            blue = bands["blue"]
            green = bands["green"]
            red = bands["red"]
            nir = bands["nir"]
            swir1 = bands["swir1"]
            swir2 = bands["swir2"]

            # reflectance_scale already applied by landsat_reflectance_stack,
            # therefore we pass reflectance_scale=None to avoid double scaling.
            reflectance_scale = None

        return self.predict_from_bands(
            blue,
            green,
            red,
            nir,
            swir1,
            swir2,
            reflectance_scale=reflectance_scale,
            water_threshold=water_threshold,
            **kwargs,
        )

    def estimate_salinity_from_mosaic(
        self,
        mosaic_path: Path,
        water_threshold: float = 0.2,
        score_path: str | None = None,
        class_path: str | None = None,
        water_path: str | None = None,
    ) -> dict | None:
        """
        Estimate salinity products from a Landsat mosaic path and write rasters.

        Returns dict with paths: {"score", "class", "water_mask"} or None on failure.
        """
        try:
            with rasterio.open(mosaic_path) as src:
                if src.count < 6:
                    logging.warning(
                        "Expected ≥6 bands for salinity estimation, found %d in %s",
                        src.count,
                        mosaic_path.name,
                    )
                    return None
                profile = src.profile
                bands = self.landsat_reflectance_stack(src)
        except RasterioError as exc:
            logging.warning("Unable to open mosaic %s: %s", mosaic_path, exc)
            return None

        required = {"blue", "green", "red", "nir", "swir1", "swir2"}
        missing = required - bands.keys()
        if missing:
            logging.warning(
                "Missing required bands %s in %s; skipping salinity estimation",
                missing,
                mosaic_path.name,
            )
            return None

        salinity = self.predict_from_bands(
            blue=bands["blue"],
            green=bands["green"],
            red=bands["red"],
            nir=bands["nir"],
            swir1=bands["swir1"],
            swir2=bands["swir2"],
            reflectance_scale=None,
            water_threshold=water_threshold,
        )

        if salinity is None:
            logging.warning("Salinity estimation skipped for %s", mosaic_path)
            return None

        base = mosaic_path.with_suffix("")
        if not score_path:
            score_path = base.with_name(f"{base.stem}_salinity_score.tif")
        if not class_path:
            class_path = base.with_name(f"{base.stem}_salinity_class.tif")
        if not water_path:
            water_path = base.with_name(f"{base.stem}_salinity_water_mask.tif")

        logging.info("Writing salinity products for %s", mosaic_path.name)

        # Write score raster
        self.write_single_band(score_path, profile, salinity["score"], dtype="float32")

        # Convert string labels → integer codes
        class_codes = np.vectorize(
            lambda v: SALINITY_CLASS_CODES.get(v, 0), otypes=[np.uint8]
        )(salinity["class_map"])
        self.write_single_band(
            class_path, profile, class_codes, dtype="uint8", nodata=255
        )

        # Write combined water mask
        self.write_single_band(
            water_path,
            profile,
            salinity["water_mask"].astype(np.float32),
            dtype="float32",
            nodata=np.nan,
        )

        return {"score": score_path, "class": class_path, "water_mask": water_path}

    # Because this is a rule-based model we don't implement a PyTorch forward that
    # expects tensors for training. However we provide a small wrapper for API
    # compatibility.
    def forward(self, x):
        """
        Forward is implemented to accept a rasterio DatasetReader, a Path to a
        mosaic, or a dict of in-memory band arrays. It dispatches to the
        appropriate helper.
        """
        # dispatch based on type
        if isinstance(x, (str, Path)):
            return self.estimate_salinity_from_mosaic(Path(x))
        if hasattr(x, "read") and callable(getattr(x, "read")):
            # rasterio DatasetReader-like
            return self.predict_from_rasterio(x)
        if isinstance(x, dict):
            expected = {"blue", "green", "red", "nir", "swir1", "swir2"}
            if not expected.issubset(set(x.keys())):
                raise ValueError(f"Band dict must contain keys: {expected}")
            return self.predict_from_bands(
                x["blue"], x["green"], x["red"], x["nir"], x["swir1"], x["swir2"]
            )

        raise ValueError("Unsupported input to SalinityHeuristicModel.forward")
