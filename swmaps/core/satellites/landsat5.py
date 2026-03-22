"""Landsat-5 Thematic Mapper (TM) mission configuration.

Landsat-5 TM was operational from 1984-03-01 to 2013-01-01, providing
30 m multispectral imagery in six reflective bands (Blue through SWIR-2).
GEE collection: ``LANDSAT/LT05/C02/T1_L2`` (Collection 2 Level-2 SR).
"""

import numpy as np
from .base import Mission


class Landsat5(Mission):
    """Landsat-5 Thematic Mapper (TM) mission.

    Band mapping (Collection 2 Level-2 surface reflectance):

    - blue   -> SR_B1 (band index 1)
    - green  -> SR_B2 (band index 2)
    - red    -> SR_B3 (band index 3)
    - nir08  -> SR_B4 (band index 4)
    - swir16 -> SR_B5 (band index 5)
    - swir22 -> SR_B7 (band index 6)

    DN-to-reflectance scale factor: ``0.0000275 * DN - 0.2``
    (Landsat Collection 2 standard).
    """

    def __init__(self) -> None:
        super().__init__("landsat-5")

    def reflectance_stack(self, src):
        return {
            "blue": src["SR_B1"].values,
            "green": src["SR_B2"].values,
            "red": src["SR_B3"].values,
            "nir08": src["SR_B4"].values,
            "swir16": src["SR_B5"].values,
            "swir22": src["SR_B7"].values,
        }

    def band_indices(self):
        return {"blue": 1, "green": 2, "red": 3, "nir08": 4, "swir16": 5, "swir22": 6}

    def bands(self):
        return {
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir08": "SR_B4",
            "swir16": "SR_B5",
            "swir22": "SR_B7",
        }

    @property
    def gee_collection(self):
        return "LANDSAT/LT05/C02/T1_L2"

    @property
    def gee_scale(self):
        return 30

    @property
    def collection(self):
        return "landsat-c2-l2"

    @property
    def resolution(self):
        return 30

    @property
    def valid_date_range(self):
        return ("1984-03-01", "2013-01-01")
    
    def read_bands(self, src):
        """Read and scale Landsat-5 Collection 2 DN values to surface reflectance.

        Applies the standard Collection 2 scale factor ``0.0000275 * DN - 0.2``
        to produce float32 arrays in the ``[0, 1]`` reflectance range.

        Args:
            src: An open :class:`rasterio.io.DatasetReader` with at least 6 bands.

        Returns:
            Dict[str, np.ndarray]: Band dict with keys ``"blue"``, ``"green"``,
            ``"red"``, ``"nir"``, ``"swir1"``, ``"swir2"`` in ``[0, 1]``.

        Raises:
            ValueError: If *src* has fewer than 6 bands.
        """
        if src.count < 6:
            raise ValueError(
                f"Expected ≥6 bands for Landsat-5, got {src.count}."
            )
        scale, offset = 0.0000275, -0.2
        return {
            "blue":  src.read(1).astype(np.float32) * scale + offset,
            "green": src.read(2).astype(np.float32) * scale + offset,
            "red":   src.read(3).astype(np.float32) * scale + offset,
            "nir":   src.read(4).astype(np.float32) * scale + offset,
            "swir1": src.read(5).astype(np.float32) * scale + offset,
            "swir2": src.read(6).astype(np.float32) * scale + offset,
        }
