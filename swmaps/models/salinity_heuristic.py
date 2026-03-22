"""Heuristic (rule-based) salinity estimation model.

The public interface is :meth:`~SalinityHeuristicModel.predict`, which accepts
a band dict and returns a result dict. No filesystem I/O happens inside the
model - all raster reading and writing lives in the pipeline layer
(:mod:`swmaps.pipeline.salinity`).

Band dict contract
------------------
Every method that takes a ``bands`` argument expects a dict with exactly these
six keys, each mapping to a float32 NumPy array of the same ``(H, W)`` shape::

    {
        "blue":   np.ndarray,  # surface reflectance [0, 1]
        "green":  np.ndarray,
        "red":    np.ndarray,
        "nir":    np.ndarray,
        "swir1":  np.ndarray,
        "swir2":  np.ndarray,
    }
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from swmaps.models.base import BaseSalinityModel

REQUIRED_BANDS = frozenset({"blue", "green", "red", "nir", "swir1", "swir2"})

# Map salinity classes to integer codes for compact raster storage
SALINITY_CLASS_CODES: dict[str, int] = {
    "land": 0,
    "fresh": 1,
    "brackish": 2,
    "saline": 3,
}

class SalinityHeuristicModel(BaseSalinityModel):
    """Heuristic (rule-based) salinity classifier.

    The sole entry point for callers is :meth:`predict`, which accepts a band
    dict and returns a result dict. All filesystem I/O is handled by the
    pipeline layer.

    This model is stateless and not trainable - :meth:`train_model` raises
    :exc:`NotImplementedError`.

    Args:
        output_dim: Retained for API compatibility with
            :class:`~swmaps.models.base.BaseSalinityModel`. Always ``1``.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__(output_dim=output_dim)

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

    def predict(
        self,
        bands: Dict[str, np.ndarray],
        *,
        water_threshold: float = 0.2,
        salinity_proxy_scale: float = 1.2,
        salinity_proxy_threshold: float = 0.35,
        chlorophyll_reference: float = 2.0,
    ) -> Dict[str, np.ndarray]:
        """Run heuristic salinity estimation on in-memory band arrays.

        This is the sole public entry point. No filesystem I/O is performed.

        Args:
            bands: Dict with keys ``"blue"``, ``"green"``, ``"red"``,
                ``"nir"``, ``"swir1"``, ``"swir2"`` as float32 arrays
                of identical shape ``(H, W)`` in ``[0, 1]``.
            water_threshold: Minimum NDWI or MNDWI value to classify a
                pixel as water. Defaults to ``0.2``.
            salinity_proxy_scale: Divisor to normalise ``swir1 + swir2``
                to ``[0, 1]``.
            salinity_proxy_threshold: Minimum normalised salinity proxy
                to expand the water mask via salinity override.
            chlorophyll_reference: Reference ratio for chlorophyll
                normalisation.

        Returns:
            Dict with keys:

            - ``"score"`` - float32 ``(H, W)``; ``NaN`` outside water mask.
            - ``"class_map"`` - string array, values ``"land"``,
              ``"fresh"``, ``"brackish"``, ``"saline"``.
            - ``"class_codes"`` - uint8 integer encoding of class_map.
            - ``"water_mask"`` - bool ``(H, W)``.
            - ``"indices"`` - dict of intermediate spectral indices.

        Raises:
            ValueError: If *bands* is missing any required key.
        """
        missing = REQUIRED_BANDS - bands.keys()
        if missing:
            raise ValueError(
                f"Band dict is missing required keys: {sorted(missing)}. "
                f"Required: {sorted(REQUIRED_BANDS)}."
            )

        b = {
            k: np.asarray(v, dtype=np.float32, order="C")
            for k, v in bands.items()
        }

        ndwi  = self._safe_normalized_difference(b["green"], b["nir"])
        mndwi = self._safe_normalized_difference(b["green"], b["swir1"])
        ndvi  = self._safe_normalized_difference(b["nir"],   b["red"])
        ndti  = self._safe_normalized_difference(b["green"], b["blue"])

        turbidity_ratio = np.divide(
            b["red"], b["green"],
            out=np.zeros_like(b["red"], dtype=np.float32),
            where=b["green"] != 0,
        ).astype(np.float32)

        chlorophyll_ratio = np.divide(
            b["green"], b["blue"],
            out=np.zeros_like(b["green"], dtype=np.float32),
            where=b["blue"] != 0,
        ).astype(np.float32)

        salinity_proxy = np.clip(
            b["swir1"] + b["swir2"], 0.0, None
        ).astype(np.float32)
        salinity_proxy_norm = np.clip(
            salinity_proxy / salinity_proxy_scale, 0.0, 1.0
        )

        chlorophyll_norm = np.clip(
            chlorophyll_ratio / chlorophyll_reference, 0.0, 1.0
        )
        turbidity_norm = np.clip(turbidity_ratio / 2.0, 0.0, 1.0)

        ndwi_scaled  = np.clip((ndwi  + 1.0) / 2.0, 0.0, 1.0)
        mndwi_scaled = np.clip((mndwi + 1.0) / 2.0, 0.0, 1.0)

        water_mask = (ndwi > water_threshold) | (mndwi > water_threshold)
        weak_water = (ndwi > 0.0) | (mndwi > 0.0)
        salinity_override = (
            salinity_proxy_norm > salinity_proxy_threshold
        ) & weak_water
        water_mask = water_mask | salinity_override

        score = (
            0.20 * (1.0 - ndwi_scaled)
            + 0.15 * (1.0 - mndwi_scaled)
            + 0.45 * salinity_proxy_norm
            + 0.10 * turbidity_norm
            + 0.10 * (1.0 - chlorophyll_norm)
        )
        score = np.clip(score, 0.0, 1.0).astype(np.float32)
        score = np.where(water_mask, score, np.nan).astype(np.float32)

        score_filled = np.nan_to_num(score, nan=0.0)
        class_map = np.where(
            water_mask,
            np.select(
                [score_filled < 0.35, score_filled < 0.60],
                ["fresh", "brackish"],
                default="saline",
            ),
            "land",
        )

        class_codes = np.vectorize(
            lambda v: SALINITY_CLASS_CODES.get(v, 0), otypes=[np.uint8]
        )(class_map)

        return {
            "score": score,
            "class_map": class_map,
            "class_codes": class_codes,
            "water_mask": water_mask,
            "indices": {
                "ndwi": ndwi,
                "mndwi": mndwi,
                "ndvi": ndvi,
                "ndti": ndti,
                "turbidity_ratio": turbidity_ratio,
                "chlorophyll_ratio": chlorophyll_ratio,
                "salinity_proxy": salinity_proxy,
                "salinity_proxy_norm": salinity_proxy_norm,
            },
        }
    
    def forward(self, bands: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """Alias of :meth:`predict` for PyTorch-style compatibility.

        Args:
            bands: Band dict - same contract as :meth:`predict`.
            **kwargs: Forwarded to :meth:`predict`.

        Returns:
            Dict: Same structure as :meth:`predict`.
        """
        return self.predict(bands, **kwargs)

    def train_model(self, *args, **kwargs):
        """Not implemented - this model is rule-based and has no trainable parameters.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SalinityHeuristicModel is rule-based and cannot be trained."
        )
