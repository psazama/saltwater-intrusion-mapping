from .base import Mission


# --------------------------
# Landsat-7
# --------------------------
class Landsat7(Mission):
    def __init__(self):
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
