"""Mission configuration helpers for Landsat and Sentinel sensors."""


def get_mission(mission: str) -> dict[str, object]:
    """Return metadata describing the requested satellite mission."""

    # ---------------------------------------------------------
    # Sentinel-2 (harmonized SR)
    # ---------------------------------------------------------
    if mission == "sentinel-2":
        gee_collection = "COPERNICUS/S2_SR_HARMONIZED"
        gee_scale = 10

        bands = {
            "blue": "B2",
            "green": "B3",
            "red": "B4",
            "nir08": "B8",
            "swir16": "B11",
            "swir22": "B12",
        }

        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }

        # Legacy STAC fields (ignored by GEE pipeline)
        collection = "sentinel-2-l2a"
        query_filter = {"eo:cloud_cover": {"lt": 10}}
        resolution = 10
        valid_date_range = ["2015-06-23", None]

    # ---------------------------------------------------------
    # Landsat-5 Collection 2 Level 2
    # ---------------------------------------------------------
    elif mission == "landsat-5":
        gee_collection = "LANDSAT/LT05/C02/T1_L2"
        gee_scale = 30

        bands = {
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir08": "SR_B4",
            "swir16": "SR_B5",
            "swir22": "SR_B7",
        }

        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }

        collection = "landsat-c2-l2"
        query_filter = {"eo:cloud_cover": {"lt": 10}}
        resolution = 30
        valid_date_range = ["1984-03-01", "2013-01-01"]

    # ---------------------------------------------------------
    # Landsat-7 Collection 2 Level 2
    # ---------------------------------------------------------
    elif mission == "landsat-7":
        gee_collection = "LANDSAT/LE07/C02/T1_L2"
        gee_scale = 30

        bands = {
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir08": "SR_B4",
            "swir16": "SR_B5",
            "swir22": "SR_B7",
        }

        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }

        collection = "landsat-c2-l2"
        query_filter = {"eo:cloud_cover": {"lt": 10}}
        resolution = 30
        valid_date_range = ["1999-04-15", "2022-03-31"]

    else:
        raise ValueError("Unsupported mission")

    return {
        "bands": bands,
        "band_index": band_index,
        "collection": collection,
        "query_filter": query_filter,
        "resolution": resolution,
        "valid_date_range": valid_date_range,
        "gee_collection": gee_collection,
        "gee_scale": gee_scale,
    }
