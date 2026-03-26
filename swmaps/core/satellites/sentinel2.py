"""Sentinel-2 MultiSpectral Instrument (MSI) mission configuration.

Sentinel-2 MSI has been operational since 2015-06-23 (Sentinel-2A launch).
At 10 m native resolution for visible and NIR bands, it provides the finest
spatial detail of the three missions currently supported by this pipeline.
GEE collection: ``COPERNICUS/S2_SR_HARMONIZED`` (harmonised Level-2A SR).
"""

import numpy as np

from .base import Mission


class Sentinel2(Mission):
    """Sentinel-2 MultiSpectral Instrument (MSI) mission.

    Band mapping (Level-2A surface reflectance, ``COPERNICUS/S2_SR_HARMONIZED``):

    - blue   -> B2  (band index 1)
    - green  -> B3  (band index 2)
    - red    -> B4  (band index 3)
    - nir08  -> B8  (band index 4)
    - swir16 -> B11 (band index 5)
    - swir22 -> B12 (band index 6)

    Sentinel-2 SR values are scaled integers - divide by 10 000 for
    reflectance in [0, 1].  The pipeline handles this via the
    ``reflectance_scale`` parameter in the salinity model.
    """

    def __init__(self) -> None:
        super().__init__("sentinel-2")

    def reflectance_stack(self, src):
        return {
            "blue": src["B2"].values,
            "green": src["B3"].values,
            "red": src["B4"].values,
            "nir08": src["B8"].values,
            "swir16": src["B11"].values,
            "swir22": src["B12"].values,
        }

    def band_indices(self):
        return {"blue": 1, "green": 2, "red": 3, "nir08": 4, "swir16": 5, "swir22": 6}

    def bands(self):
        return {
            "blue": "B2",
            "green": "B3",
            "red": "B4",
            "nir08": "B8",
            "swir16": "B11",
            "swir22": "B12",
        }

    @property
    def gee_collection(self):
        return "COPERNICUS/S2_SR_HARMONIZED"

    @property
    def gee_scale(self):
        return 10

    @property
    def collection(self):
        return "sentinel-2-l2a"

    @property
    def resolution(self):
        return 10

    @property
    def valid_date_range(self):
        return ("2015-06-23", None)

    def read_bands(self, src):
        """Read and scale Sentinel-2 SR values to surface reflectance.

        Sentinel-2 Level-2A values are scaled integers in the range 0–10 000.
        This method divides by 10 000 to produce float32 arrays in ``[0, 1]``.

        Args:
            src: An open :class:`rasterio.io.DatasetReader` with at least 6 bands.

        Returns:
            Dict[str, np.ndarray]: Band dict with keys ``"blue"``, ``"green"``,
            ``"red"``, ``"nir"``, ``"swir1"``, ``"swir2"`` in ``[0, 1]``.

        Raises:
            ValueError: If *src* has fewer than 6 bands.
        """
        if src.count < 6:
            raise ValueError(f"Expected ≥6 bands for Sentinel-2, got {src.count}.")
        scale = 10_000.0
        return {
            "blue": src.read(1).astype(np.float32) / scale,
            "green": src.read(2).astype(np.float32) / scale,
            "red": src.read(3).astype(np.float32) / scale,
            "nir": src.read(4).astype(np.float32) / scale,
            "swir1": src.read(5).astype(np.float32) / scale,
            "swir2": src.read(6).astype(np.float32) / scale,
        }
