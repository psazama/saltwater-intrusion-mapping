"""Shared types used across the swmaps models and pipeline layers.

This module defines the two foundational types that flow through the entire
pipeline:

- :class:`PipelineResult` - a structured return value for every pipeline
  function, carrying status, produced file paths, and any error message.
  Designed to serialise cleanly to JSON for a Flask/FastAPI response.

- Pydantic config models (``DownloadConfig``, ``SegmentationConfig``,
  ``SalinityConfig``, ``TrendConfig``, ``WorkflowConfig``) - typed
  representations of the TOML workflow configuration.  Each exposes a
  :meth:`from_dict` classmethod so that TOML-loaded dicts and API request
  bodies can both be validated through the same path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Structured return value for every pipeline function.

    All pipeline functions return a ``PipelineResult`` so callers - whether
    the workflow runner, a Flask view, or a FastAPI endpoint - have a uniform
    object to inspect, log, and serialise.

    Attributes:
        status: One of ``"ok"``, ``"skipped"``, or ``"error"``.
        output_paths: List of :class:`~pathlib.Path` objects produced by the
            pipeline step.  Empty when *status* is ``"skipped"`` or
            ``"error"``.
        error: Human-readable error message.  ``None`` when *status* is
            ``"ok"`` or ``"skipped"``.
        meta: Optional dict of supplementary information (e.g. scene counts,
            timing).  Safe to include in JSON responses.
    """

    status: str  # "ok" | "skipped" | "error"
    output_paths: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def ok(cls, paths: List[Path], **meta) -> "PipelineResult":
        """Create a successful result.

        Args:
            paths: Files produced by the pipeline step.
            **meta: Arbitrary key-value pairs stored in :attr:`meta`.

        Returns:
            PipelineResult: Result with ``status="ok"``.
        """
        return cls(status="ok", output_paths=list(paths), meta=dict(meta))

    @classmethod
    def skipped(cls, reason: str = "", **meta) -> "PipelineResult":
        """Create a skipped result (step was a no-op, e.g. output already exists).

        Args:
            reason: Short human-readable explanation.
            **meta: Arbitrary key-value pairs stored in :attr:`meta`.

        Returns:
            PipelineResult: Result with ``status="skipped"``.
        """
        return cls(status="skipped", error=reason or None, meta=dict(meta))

    @classmethod
    def failure(cls, message: str, **meta) -> "PipelineResult":
        """Create an error result.

        Args:
            message: Description of what went wrong.
            **meta: Arbitrary key-value pairs stored in :attr:`meta`.

        Returns:
            PipelineResult: Result with ``status="error"``.
        """
        return cls(status="error", error=message, meta=dict(meta))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON responses.

        Returns:
            dict: Keys ``status``, ``output_paths`` (list of str),
            ``error``, and ``meta``.
        """
        return {
            "status": self.status,
            "output_paths": [str(p) for p in self.output_paths],
            "error": self.error,
            "meta": self.meta,
        }

    @property
    def is_ok(self) -> bool:
        """``True`` when the step completed without error."""
        return self.status == "ok"


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class DownloadConfig(BaseModel):
    """Typed configuration for the imagery download pipeline step.

    Covers all fields read by :func:`~swmaps.pipeline.download.download_data`.
    Supports both explicit coordinate lists and GeoJSON geometry paths.

    Args:
        start_date: First date in the acquisition window (ISO-8601).
        end_date: Last date in the acquisition window (ISO-8601).
        mission: One or more satellite mission slugs, e.g.
            ``["sentinel-2", "landsat-5"]``.
        out_dir: Root directory where downloaded mosaics are written.
        geometry: Path to a GeoJSON defining the AOI centroid.  Mutually
            exclusive with *latitude* / *longitude*.
        latitude: Centre latitude(s).  Scalar or list.
        longitude: Centre longitude(s).  Scalar or list.
        date_step: Step in days between sampled acquisition dates.
        buffer_km: Bounding-box half-width around each centre point, km.
        cloud_filter: Maximum allowed cloud cover percentage (0–100).
        days_before: Temporal window before each target date, days.
        days_after: Temporal window after each target date, days.
        samples_per_date: Maximum number of scenes to return per date step.
        save_png: Whether to write RGB preview PNGs alongside GeoTIFFs.
        skip_download: When ``True``, skip the download step entirely.
        download_cdl: Whether to download the USDA NASS CDL alongside imagery.
        cdl_year: CDL product year.
        cdl_region: AOI for CDL download - GeoJSON path or
            ``[xmin, ymin, xmax, ymax]``.
        cdl_buffer_km: Buffer around lat/lon used when *cdl_region* is absent.
        cdl_out: Destination path for the CDL GeoTIFF.

    Example::

        cfg = DownloadConfig(
            start_date="2021-01-01",
            end_date="2021-12-31",
            mission=["sentinel-2"],
            geometry="config/choptank_river_region.geojson",
        )
    """

    start_date: str
    end_date: str
    mission: Union[List[str], str] = Field(default="sentinel-2")
    out_dir: Optional[str] = None

    # Location - one of geometry OR lat/lon
    geometry: Optional[str] = None
    latitude: Optional[Union[float, List[float]]] = None
    longitude: Optional[Union[float, List[float]]] = None

    # Temporal / spatial
    date_step: int = 1
    buffer_km: float = 1.0
    cloud_filter: float = 30.0
    days_before: int = 7
    days_after: int = 7
    samples_per_date: int = 1
    save_png: bool = False
    skip_download: bool = False

    # CDL options
    download_cdl: bool = False
    cdl_year: int = 2019
    cdl_region: Optional[Union[str, List[float]]] = None
    cdl_buffer_km: float = 1.0
    cdl_out: Optional[str] = None

    @field_validator("mission", mode="before")
    @classmethod
    def normalise_mission(cls, v):
        """Accept a scalar string or comma-separated string as a list."""
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    @model_validator(mode="after")
    def require_location(self):
        """Ensure either *geometry* or *latitude*/*longitude* is provided."""
        if not self.geometry and (self.latitude is None or self.longitude is None):
            raise ValueError(
                "Supply either 'geometry' (GeoJSON path) or both 'latitude' "
                "and 'longitude'."
            )
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "DownloadConfig":
        """Construct from a raw TOML-loaded dict, ignoring unknown keys.

        Args:
            d: Dict as returned by ``tomllib.load()``.

        Returns:
            DownloadConfig: Validated config instance.
        """
        known = cls.model_fields.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


class SegmentationConfig(BaseModel):
    """Typed configuration for segmentation training and inference.

    Args:
        model_type: Architecture name - ``"farseg"`` or ``"panopticon"``.
        segmentation_out_dir: Directory where prediction rasters are written.
        segmentation_model_dir: Directory for checkpoints and training logs.
        segmentation_weights_path: Path to a pre-trained ``.pth`` checkpoint
            for inference.
        segmentation_num_classes: Number of output classes for the model head.
        train_segmentation: Whether to run model training.
        run_segmentation: Whether to run inference after training.
        segmentation_png: Save RGB PNG previews of predictions.
        epochs: Training epochs.
        batch_size: Samples per mini-batch.
        learning_rate: Initial Adam learning rate.
        loss: Loss function name - ``"ce"``, ``"focal"``, ``"dice"``,
            ``"hybrid"``.
        do_val: Include a validation split during training.
        val_dir: Directory containing validation imagery.
        val_start_date: Start of the validation date range (ISO-8601).
        val_end_date: End of the validation date range (ISO-8601).
        val_region: GeoJSON path for the validation AOI.
    """

    model_type: str = "farseg"
    segmentation_out_dir: Optional[str] = None
    segmentation_model_dir: Optional[str] = None
    segmentation_weights_path: Optional[str] = None
    segmentation_num_classes: int = 256
    train_segmentation: bool = False
    run_segmentation: bool = False
    segmentation_png: bool = False

    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    loss: str = "dice"

    # Validation
    do_val: bool = False
    val_dir: Optional[str] = None
    val_start_date: Optional[str] = None
    val_end_date: Optional[str] = None
    val_region: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationConfig":
        """Construct from a raw TOML-loaded dict, ignoring unknown keys.

        Args:
            d: Dict as returned by ``tomllib.load()``.

        Returns:
            SegmentationConfig: Validated config instance.
        """
        known = cls.model_fields.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


class SalinityConfig(BaseModel):
    """Typed configuration for the salinity ground-truth and classification steps.

    Args:
        run_salinity_pipeline: Run the full ground-truth download + match step.
        run_salinity_classification: Run per-mosaic heuristic salinity
            classification.
        truth_download_list: Text file listing CODC NetCDF filenames to
            download.
        truth_dir: Directory containing (or where to save) CODC ``.nc`` files.
        truth_file: Output CSV path for the combined salinity truth table.
        water_threshold: NDWI threshold for water/land separation in the
            heuristic model.
    """

    run_salinity_pipeline: bool = False
    run_salinity_classification: bool = False
    truth_download_list: Optional[str] = None
    truth_dir: Optional[str] = None
    truth_file: Optional[str] = None
    water_threshold: float = 0.2

    @classmethod
    def from_dict(cls, d: dict) -> "SalinityConfig":
        """Construct from a raw TOML-loaded dict, ignoring unknown keys."""
        known = cls.model_fields.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


class TrendConfig(BaseModel):
    """Typed configuration for the water-trend heatmap step.

    Args:
        run_water_trend: Whether to run mask generation and trend analysis.
        trend_output_dir: Directory to search for masks and save trend outputs.
        trend_class_value: Pixel value to track across years (default ``1``).
        trend_class_name: Substring filter for mask filenames
            (e.g. ``"mask"`` or ``"segmentation"``).
    """

    run_water_trend: bool = False
    trend_output_dir: Optional[str] = None
    trend_class_value: int = 1
    trend_class_name: str = "mask"

    @classmethod
    def from_dict(cls, d: dict) -> "TrendConfig":
        """Construct from a raw TOML-loaded dict, ignoring unknown keys."""
        known = cls.model_fields.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


class WorkflowConfig(BaseModel):
    """Top-level config that composes all pipeline step configs.

    This is the single object passed to the workflow runner (and exposed
    directly as the API request body).  Each pipeline step extracts its own
    typed sub-config from the same flat field namespace.

    Args:
        download: Download step configuration.
        segmentation: Segmentation training/inference configuration.
        salinity: Salinity pipeline configuration.
        trend: Trend analysis configuration.

    Example - build from TOML::

        import tomllib
        from swmaps.types import WorkflowConfig

        with open("examples/quickstart_train.toml", "rb") as f:
            raw = tomllib.load(f)

        cfg = WorkflowConfig.from_dict(raw)
        # cfg.download.mission  -> ["sentinel-2"]
        # cfg.segmentation.epochs -> 10

    Example - build for API::

        cfg = WorkflowConfig(
            download=DownloadConfig(
                start_date="2021-01-01",
                end_date="2021-12-31",
                mission=["sentinel-2"],
                geometry="config/choptank_river_region.geojson",
            ),
            segmentation=SegmentationConfig(run_segmentation=True),
        )
    """

    download: DownloadConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    salinity: SalinityConfig = Field(default_factory=SalinityConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowConfig":
        """Build a ``WorkflowConfig`` from a flat TOML-loaded dict.

        Each sub-config is constructed by forwarding the full dict and letting
        each model's ``from_dict`` ignore unknown keys.  This means a single
        TOML file can carry all fields with no nesting required.

        Args:
            d: Flat dict as returned by ``tomllib.load()``.

        Returns:
            WorkflowConfig: Fully validated workflow configuration.

        Raises:
            pydantic.ValidationError: If required fields are missing or values
                fail validation.
        """
        return cls(
            download=DownloadConfig.from_dict(d),
            segmentation=SegmentationConfig.from_dict(d),
            salinity=SalinityConfig.from_dict(d),
            trend=TrendConfig.from_dict(d),
        )


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class SceneResponse(BaseModel):
    """API response model for a single imagery scene.

    Attributes:
        scene_id: GEE scene identifier.
        sensor: Mission slug, e.g. ``"sentinel-2"``.
        acquisition_date: ISO-8601 acquisition date.
        band_count: Number of spectral bands.
        crs: Coordinate reference system string.
        status: One of ``"active"``, ``"missing"``, ``"archived"``.
        file_locations: List of file paths or ``gs://`` URIs.
        ingest_timestamp: When the scene was registered.
        version_no: Incremented when the file hash changes on re-ingest.
    """

    scene_id: str
    sensor: str
    acquisition_date: str
    band_count: Optional[int] = None
    crs: Optional[str] = None
    status: str
    file_locations: List[str]
    ingest_timestamp: Optional[str] = None
    version_no: int = 1
    location_wkt: Optional[str] = None

    @classmethod
    def from_row(cls, row: dict) -> SceneResponse:
        """Construct from a psycopg2 RealDictRow.

        Args:
            row: Database row dict from :func:`~swmaps.infra.db.fetch_scene`.

        Returns:
            SceneResponse: Validated response model instance.
        """
        return cls(
            scene_id=row["scene_id"],
            sensor=row["sensor"],
            acquisition_date=str(row["acquisition_date"]),
            band_count=row.get("band_count"),
            crs=row.get("crs"),
            status=row["status"],
            file_locations=row["file_locations"],
            ingest_timestamp=(
                str(row["ingest_timestamp"]) if row.get("ingest_timestamp") else None
            ),
            version_no=row.get("version_no", 1),
            location_wkt=row.get("location_wkt"),
        )


class SalinityProfileResponse(BaseModel):
    """API response model for a single salinity profile.

    Attributes:
        cast_id: Unique cast identifier.
        sample_date: ISO-8601 sample date.
        surface_salinity: Near-surface salinity in PSU.
        max_depth: Maximum sampling depth in metres.
        source_file: Source NetCDF filename.
        ingested_at: When the profile was registered.
    """

    cast_id: str
    sample_date: str
    surface_salinity: float
    max_depth: Optional[float] = None
    source_file: str
    ingested_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: dict) -> SalinityProfileResponse:
        """Construct from a psycopg2 RealDictRow.

        Args:
            row: Database row dict from
                :func:`~swmaps.infra.db.fetch_salinity_profile`.

        Returns:
            SalinityProfileResponse: Validated response model instance.
        """
        return cls(
            cast_id=row["cast_id"],
            sample_date=str(row["sample_date"]),
            surface_salinity=row["surface_salinity"],
            max_depth=row.get("max_depth"),
            source_file=row["source_file"],
            ingested_at=str(row["ingested_at"]) if row.get("ingested_at") else None,
        )


class DepthProfileResponse(BaseModel):
    """API response model for a single depth level in a salinity cast.

    Attributes:
        cast_id: Parent cast identifier.
        depth_m: Sampling depth in metres.
        salinity: Salinity observation in PSU.
        temperature: Temperature observation in degrees Celsius.
    """

    cast_id: str
    depth_m: float
    salinity: Optional[float] = None
    temperature: Optional[float] = None

    @classmethod
    def from_row(cls, row: dict) -> DepthProfileResponse:
        """Construct from a psycopg2 RealDictRow.

        Args:
            row: Database row dict from
                :func:`~swmaps.infra.db.fetch_depth_profile`.

        Returns:
            DepthProfileResponse: Validated response model instance.
        """
        return cls(
            cast_id=row["cast_id"],
            depth_m=row["depth_m"],
            salinity=row.get("salinity"),
            temperature=row.get("temperature"),
        )


class ProcessingRunResponse(BaseModel):
    """API response model for a single processing run.

    Attributes:
        product_id: Unique product identifier.
        base_scene_id: Scene this product was derived from.
        task: Pipeline task name, e.g. ``"water_mask"``.
        status: One of ``"not_started"``, ``"running"``, ``"complete"``,
            ``"failed"``.
        started_at: When the run was registered.
        completed_at: When the run finished, or ``None`` if still running.
        output_paths: List of output file paths or ``gs://`` URIs.
        error_message: Error description when status is ``"failed"``.
        parameters: Task parameters used for this run.
    """

    product_id: str
    base_scene_id: str
    task: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_paths: Optional[List[str]] = None
    error_message: Optional[str] = None
    parameters: Optional[dict] = None

    @classmethod
    def from_row(cls, row: dict) -> ProcessingRunResponse:
        """Construct from a psycopg2 RealDictRow.

        Args:
            row: Database row dict from
                :func:`~swmaps.infra.db.fetch_processing_run`.

        Returns:
            ProcessingRunResponse: Validated response model instance.
        """
        return cls(
            product_id=row["product_id"],
            base_scene_id=row["base_scene_id"],
            task=row["task"],
            status=row["status"],
            started_at=str(row["started_at"]) if row.get("started_at") else None,
            completed_at=str(row["completed_at"]) if row.get("completed_at") else None,
            output_paths=row.get("output_paths"),
            error_message=row.get("error_message"),
            parameters=row.get("parameters"),
        )
