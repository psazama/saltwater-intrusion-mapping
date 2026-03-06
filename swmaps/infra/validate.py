import re
import tomllib
from datetime import date
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

with open("config/validation.toml", "rb") as f:
    config = tomllib.load(f)

STUDY_AREA_BBOX = config["validation"]
KNOWN_SENSORS = config["validation"]["known_sensors"]


class ImageryRecord(BaseModel):
    scene_id: str
    location_wkt: str
    band_count: int
    acquisition_date: date
    sensor: str
    file_locations: list[str]
    crs: Optional[str] = None

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, v):
        # Check that scene_id contains at least a date-like substring
        # Matches YYYYMMDD or YYYY-MM-DD anywhere in the string
        date_pattern = re.compile(r"\d{8}|\d{4}-\d{2}-\d{2}")
        if not date_pattern.search(v):
            raise ValueError(f"scene_id '{v}' does not appear to contain a valid date")
        return v

    @field_validator("acquisition_date")
    @classmethod
    def validate_acquisition_date(cls, v):
        if v > date.today():
            raise ValueError(f"acquisition_date {v} is in the future")
        return v

    @field_validator("band_count")
    @classmethod
    def validate_band_count(cls, v):
        if v <= 0:
            raise ValueError(f"band_count must be a positive integer, got {v}")
        return v

    @field_validator("file_locations")
    @classmethod
    def validate_file_locations(cls, v):
        if len(v) == 0:
            raise ValueError("file_locations cannot be empty")
        return v

    @field_validator("sensor")
    @classmethod
    def validate_sensor(cls, v):
        if v.lower() not in KNOWN_SENSORS:
            raise ValueError(f"Unknown sensor '{v}', expected one of {KNOWN_SENSORS}")
        return v.lower()

    @model_validator(mode="after")
    def validate_location_in_study_area(self):
        # Parse the WKT polygon and check all coordinates are within bbox
        # Extracts coordinate pairs from WKT string
        coords_str = re.findall(r"-?\d+\.?\d*\s-?\d+\.?\d*", self.location_wkt)
        for coord in coords_str:
            lon, lat = map(float, coord.split())
            if not (STUDY_AREA_BBOX["min_lon"] <= lon <= STUDY_AREA_BBOX["max_lon"]):
                raise ValueError(f"Latitude {lat} is outside study area")
        return self
