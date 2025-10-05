"""Mission configuration helpers for Landsat and Sentinel sensors."""


def get_mission(mission: str) -> dict[str, object]:
    """Return metadata describing the requested satellite mission.

    Args:
        mission (str): Mission slug such as ``"sentinel-2"`` or
            ``"landsat-5"``.

    Returns:
        dict[str, object]: Dictionary containing band mappings, band indices,
        collection identifiers, query filters, resolution, and valid date
        range.

    Raises:
        ValueError: If the mission slug is not recognised.
    """
    if mission == "sentinel-2":
        collection = "sentinel-2-l2a"
        query_filter = {"eo:cloud_cover": {"lt": 10}}
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2",
        }
        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }
        resolution = 10
        valid_date_range = ["2015-06-23", None]
    elif mission == "landsat-5":
        collection = "landsat-c2-l2"
        query_filter = {"eo:cloud_cover": {"lt": 10}, "platform": {"eq": "landsat-5"}}
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2",
        }
        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }
        resolution = 30
        valid_date_range = ["1984-03-01", "2013-01-01"]
    elif mission == "landsat-7":
        collection = "landsat-c2-l2"
        query_filter = {"eo:cloud_cover": {"lt": 10}, "platform": {"eq": "landsat-7"}}
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2",
        }
        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }
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
    }
