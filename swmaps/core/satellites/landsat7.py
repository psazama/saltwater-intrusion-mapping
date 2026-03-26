"""Landsat-7 Enhanced Thematic Mapper Plus (ETM+) mission configuration.

Landsat-7 ETM+ was operational from 1999-04-15 to 2022-03-31.  Note
that the Scan Line Corrector (SLC) failed in May 2003, introducing data
gaps in imagery acquired after that date.
GEE collection: ``LANDSAT/LE07/C02/T1_L2`` (Collection 2 Level-2 SR).
"""

import numpy as np

from .base import Mission


class Landsat7(Mission):
    """Landsat-7 Enhanced Thematic Mapper Plus (ETM+) mission.

    Band layout is identical to Landsat-5 - the difference lies in the
    GEE collection ID and valid date range.

    .. warning::
        Imagery acquired after 2003-05-31 contains SLC-off scan gaps.
        Downstream masking steps should account for this when processing
        post-2003 Landsat-7 tiles.

    DN-to-reflectance scale factor: ``0.0000275 * DN - 0.2``
    (Landsat Collection 2 standard).
    """

    def __init__(self) -> None:
        super().__init__("landsat-7")

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
        return "LANDSAT/LE07/C02/T1_L2"

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
        return ("1999-04-15", "2022-03-31")

    def read_bands(self, src):
        """Read and scale Landsat-7 Collection 2 DN values to surface reflectance.

        Applies the standard Collection 2 scale factor ``0.0000275 * DN - 0.2``
        to produce float32 arrays in the ``[0, 1]`` reflectance range.

        .. warning::
            Imagery acquired after 2003-05-31 contains SLC-off scan gaps
            which will appear as zero-value stripes in the output arrays.

        Args:
            src: An open :class:`rasterio.io.DatasetReader` with at least 6 bands.

        Returns:
            Dict[str, np.ndarray]: Band dict with keys ``"blue"``, ``"green"``,
            ``"red"``, ``"nir"``, ``"swir1"``, ``"swir2"`` in ``[0, 1]``.

        Raises:
            ValueError: If *src* has fewer than 6 bands.
        """
        if src.count < 6:
            raise ValueError(f"Expected ≥6 bands for Landsat-7, got {src.count}.")
        scale, offset = 0.0000275, -0.2
        return {
            "blue": src.read(1).astype(np.float32) * scale + offset,
            "green": src.read(2).astype(np.float32) * scale + offset,
            "red": src.read(3).astype(np.float32) * scale + offset,
            "nir": src.read(4).astype(np.float32) * scale + offset,
            "swir1": src.read(5).astype(np.float32) * scale + offset,
            "swir2": src.read(6).astype(np.float32) * scale + offset,
        }
