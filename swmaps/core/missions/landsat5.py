from ..missions import Mission


# --------------------------
# Landsat-5
# --------------------------
class Landsat5(Mission):
    def __init__(self):
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
