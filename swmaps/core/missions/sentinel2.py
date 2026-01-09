from ..missions import Mission


# --------------------------
# Sentinel-2
# --------------------------
class Sentinel2(Mission):
    def __init__(self):
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
