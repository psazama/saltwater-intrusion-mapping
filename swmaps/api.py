"""FastAPI application for the swmaps pipeline and catalog query layer.

This module exposes two groups of endpoints:

**Query endpoints** - read-only access to the imagery catalog, salinity
profiles, and processing run history. These wrap the db.py query functions
and return typed response models.

**Pipeline endpoints** - trigger pipeline steps by POSTing a typed config.
Each returns a :class:`~swmaps.schema.PipelineResult` serialised to JSON.

Running locally::

    uvicorn swmaps.api:app --reload

Swagger UI is available at ``http://localhost:8000/docs``.
Redoc is available at ``http://localhost:8000/redoc``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from swmaps.infra.db import (
    fetch_depth_profile,
    fetch_imagery_near_sample,
    fetch_processing_run,
    fetch_processing_runs,
    fetch_salinity_profile,
    fetch_salinity_profiles,
    fetch_scene,
    fetch_scene_products,
    fetch_scenes,
    get_connection,
)
from swmaps.pipeline.download import run_download
from swmaps.pipeline.masks import run_water_masks
from swmaps.pipeline.salinity import run_salinity_classification, run_salinity_pipeline
from swmaps.pipeline.trend import run_trend_heatmap
from swmaps.schema import (
    DepthProfileResponse,
    DownloadConfig,
    ProcessingRunResponse,
    SalinityConfig,
    SalinityProfileResponse,
    SceneResponse,
    TrendConfig,
    WorkflowConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown logic."""
    logger.info("swmaps API starting up")
    yield
    logger.info("swmaps API shutting down")


app = FastAPI(
    title="swmaps API",
    description=(
        "Query the saltwater intrusion mapping imagery catalog, salinity profiles, "
        "and processing run history. Trigger pipeline steps via typed config payloads."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn():
    """Open a database connection, raising 503 if unavailable."""
    try:
        return get_connection()
    except Exception as exc:
        logger.exception("Database connection failed")
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {exc}",
        )


def _require_spatial(
    bbox: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    radius_km: Optional[float],
) -> dict:
    """Validate and return spatial parameters as a dict.

    Accepts either a bbox string ``"min_lon,min_lat,max_lon,max_lat"`` or
    a lat/lon/radius_km combination.

    Args:
        bbox: Comma-separated bounding box string.
        lat: Center latitude.
        lon: Center longitude.
        radius_km: Search radius in kilometres.

    Returns:
        dict: Kwargs ready to unpack into a db query function.

    Raises:
        HTTPException: If neither spatial input is fully provided.
    """
    if bbox:
        try:
            parts = [float(x) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            return {"bbox": tuple(parts)}
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'",
            )

    if lat is not None and lon is not None and radius_km is not None:
        return {"lat": lat, "lon": lon, "radius_km": radius_km}

    raise HTTPException(
        status_code=422,
        detail="Provide either 'bbox' or all of 'lat', 'lon', and 'radius_km'.",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["status"])
def health() -> dict:
    """Check that the API is running.

    Returns:
        dict: ``{"status": "ok"}``
    """
    return {"status": "ok"}


@app.get("/tasks", tags=["status"])
def list_tasks() -> dict:
    """List all registered pipeline tasks.

    Returns:
        dict: Task names and their descriptions.
    """
    from swmaps.pipeline.registry import task_dict

    return {"tasks": list(task_dict.keys())}


# ---------------------------------------------------------------------------
# Scene query endpoints
# ---------------------------------------------------------------------------


@app.get("/scenes", tags=["scenes"], response_model=list[SceneResponse])
def get_scenes(
    bbox: Optional[str] = Query(
        None,
        description="Bounding box as 'min_lon,min_lat,max_lon,max_lat'",
        example="-76.0,38.0,-75.0,39.0",
    ),
    lat: Optional[float] = Query(None, description="Center latitude"),
    lon: Optional[float] = Query(None, description="Center longitude"),
    radius_km: Optional[float] = Query(None, description="Search radius in km"),
    sensor: Optional[str] = Query(
        None,
        description="Mission slug e.g. 'sentinel-2', 'landsat-5', 'landsat-7'",
    ),
    date_from: Optional[str] = Query(
        None, description="Start date ISO-8601 e.g. '2020-01-01'"
    ),
    date_to: Optional[str] = Query(
        None, description="End date ISO-8601 e.g. '2021-12-31'"
    ),
    status: str = Query("active", description="Scene status filter"),
) -> list[SceneResponse]:
    """Query imagery scenes by spatial extent and optional filters.

    Requires either *bbox* or *lat*/*lon*/*radius_km*.
    """
    spatial = _require_spatial(bbox, lat, lon, radius_km)
    with _get_conn() as conn:
        rows = fetch_scenes(
            conn,
            **spatial,
            sensor=sensor,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )
    return [SceneResponse.from_row(r) for r in rows]


@app.get("/scenes/{scene_id}", tags=["scenes"], response_model=SceneResponse)
def get_scene(scene_id: str) -> SceneResponse:
    """Fetch a single imagery scene by its scene ID.

    Args:
        scene_id: GEE scene identifier.
    """
    with _get_conn() as conn:
        row = fetch_scene(conn, scene_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    return SceneResponse.from_row(row)


@app.get(
    "/scenes/{scene_id}/products",
    tags=["scenes"],
    response_model=list[ProcessingRunResponse],
)
def get_scene_products(scene_id: str) -> list[ProcessingRunResponse]:
    """Fetch all processed products for a given scene.

    Args:
        scene_id: GEE scene identifier.
    """
    with _get_conn() as conn:
        rows = fetch_scene_products(conn, scene_id)
    return [ProcessingRunResponse.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Salinity query endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/salinity/profiles",
    tags=["salinity"],
    response_model=list[SalinityProfileResponse],
)
def get_salinity_profiles(
    bbox: Optional[str] = Query(
        None,
        description="Bounding box as 'min_lon,min_lat,max_lon,max_lat'",
    ),
    lat: Optional[float] = Query(None, description="Center latitude"),
    lon: Optional[float] = Query(None, description="Center longitude"),
    radius_km: Optional[float] = Query(None, description="Search radius in km"),
    date_from: Optional[str] = Query(None, description="Start date ISO-8601"),
    date_to: Optional[str] = Query(None, description="End date ISO-8601"),
    min_salinity: Optional[float] = Query(
        None, description="Minimum surface salinity in PSU"
    ),
    max_salinity: Optional[float] = Query(
        None, description="Maximum surface salinity in PSU"
    ),
) -> list[SalinityProfileResponse]:
    """Query salinity profiles by spatial extent and optional filters.

    Requires either *bbox* or *lat*/*lon*/*radius_km*.
    """
    spatial = _require_spatial(bbox, lat, lon, radius_km)
    with _get_conn() as conn:
        rows = fetch_salinity_profiles(
            conn,
            **spatial,
            date_from=date_from,
            date_to=date_to,
            min_salinity=min_salinity,
            max_salinity=max_salinity,
        )
    return [SalinityProfileResponse.from_row(r) for r in rows]


@app.get(
    "/salinity/profiles/{cast_id}",
    tags=["salinity"],
    response_model=SalinityProfileResponse,
)
def get_salinity_profile(cast_id: str) -> SalinityProfileResponse:
    """Fetch a single salinity profile by cast ID.

    Args:
        cast_id: Unique cast identifier.
    """
    with _get_conn() as conn:
        row = fetch_salinity_profile(conn, cast_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Cast '{cast_id}' not found.")
    return SalinityProfileResponse.from_row(row)


@app.get(
    "/salinity/profiles/{cast_id}/depth",
    tags=["salinity"],
    response_model=list[DepthProfileResponse],
)
def get_depth_profile(cast_id: str) -> list[DepthProfileResponse]:
    """Fetch all depth levels for a salinity cast.

    Args:
        cast_id: Unique cast identifier.
    """
    with _get_conn() as conn:
        rows = fetch_depth_profile(conn, cast_id)
    return [DepthProfileResponse.from_row(r) for r in rows]


@app.get(
    "/salinity/profiles/{cast_id}/imagery",
    tags=["salinity"],
    response_model=list[SceneResponse],
)
def get_imagery_near_cast(
    cast_id: str,
    radius_km: float = Query(50.0, description="Search radius in km"),
    days_window: int = Query(30, description="Days before and after sample date"),
) -> list[SceneResponse]:
    """Find imagery that spatially and temporally overlaps a salinity cast.

    Args:
        cast_id: Unique cast identifier.
        radius_km: Search radius around the cast location in km.
        days_window: Number of days before and after the sample date.
    """
    with _get_conn() as conn:
        rows = fetch_imagery_near_sample(
            conn, cast_id, radius_km=radius_km, days_window=days_window
        )
    return [SceneResponse.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Processing run query endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/runs",
    tags=["runs"],
    response_model=list[ProcessingRunResponse],
)
def get_processing_runs(
    task: Optional[str] = Query(None, description="Filter by task name"),
    status: Optional[str] = Query(None, description="Filter by status"),
) -> list[ProcessingRunResponse]:
    """List processing runs with optional task and status filters."""
    with _get_conn() as conn:
        rows = fetch_processing_runs(conn, task=task, status=status)
    return [ProcessingRunResponse.from_row(r) for r in rows]


@app.get(
    "/runs/{product_id}",
    tags=["runs"],
    response_model=ProcessingRunResponse,
)
def get_processing_run(product_id: str) -> ProcessingRunResponse:
    """Fetch a single processing run by product ID.

    Args:
        product_id: Product identifier from :func:`~swmaps.infra.db.register_processing_run`.
    """
    with _get_conn() as conn:
        row = fetch_processing_run(conn, product_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{product_id}' not found.")
    return ProcessingRunResponse.from_row(row)


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------


@app.post("/run/download", tags=["pipeline"])
def trigger_download(cfg: DownloadConfig) -> JSONResponse:
    """Trigger the imagery download pipeline step.

    Posts a :class:`~swmaps.schema.DownloadConfig` and returns a
    :class:`~swmaps.schema.PipelineResult`.
    """
    result = run_download(cfg)
    return JSONResponse(content=result.to_dict())


@app.post("/run/masks", tags=["pipeline"])
def trigger_masks(
    input_dir: str = Query(..., description="Directory of mosaics to process"),
) -> JSONResponse:
    """Trigger water mask generation for a directory of mosaics.

    Returns a :class:`~swmaps.schema.PipelineResult`.
    """
    with _get_conn() as conn:
        result = run_water_masks(Path(input_dir), conn=conn)
    return JSONResponse(content=result.to_dict())


@app.post("/run/salinity", tags=["pipeline"])
def trigger_salinity(cfg: SalinityConfig) -> JSONResponse:
    """Trigger the salinity ground-truth pipeline.

    Posts a :class:`~swmaps.schema.SalinityConfig` and returns a
    :class:`~swmaps.schema.PipelineResult`.
    """
    result = run_salinity_pipeline(cfg)
    return JSONResponse(content=result.to_dict())


@app.post("/run/salinity/classify", tags=["pipeline"])
def trigger_salinity_classify(
    cfg: SalinityConfig,
    input_dir: str = Query(..., description="Directory of mosaics to classify"),
) -> JSONResponse:
    """Trigger per-mosaic salinity classification.

    Posts a :class:`~swmaps.schema.SalinityConfig` and returns a
    :class:`~swmaps.schema.PipelineResult`.
    """
    with _get_conn() as conn:
        result = run_salinity_classification(cfg, Path(input_dir), conn=conn)
    return JSONResponse(content=result.to_dict())


@app.post("/run/trend", tags=["pipeline"])
def trigger_trend(cfg: TrendConfig) -> JSONResponse:
    """Trigger the water-trend heatmap pipeline step.

    Posts a :class:`~swmaps.schema.TrendConfig` and returns a
    :class:`~swmaps.schema.PipelineResult`.
    """
    result = run_trend_heatmap(cfg)
    return JSONResponse(content=result.to_dict())


@app.post("/run/workflow", tags=["pipeline"])
def trigger_workflow(cfg: WorkflowConfig) -> JSONResponse:
    """Trigger the full end-to-end workflow.

    Posts a :class:`~swmaps.schema.WorkflowConfig` and returns a
    dict of :class:`~swmaps.schema.PipelineResult` objects keyed by
    stage name.
    """
    results = {}

    results["download"] = run_download(cfg.download).to_dict()

    results["salinity_pipeline"] = run_salinity_pipeline(cfg.salinity).to_dict()

    with _get_conn() as conn:
        results["salinity_classification"] = run_salinity_classification(
            cfg.salinity, Path(cfg.download.out_dir or "data/outputs"), conn=conn
        ).to_dict()
        results["water_masks"] = run_water_masks(
            Path(cfg.download.out_dir or "data/outputs"),
            conn=conn,
        ).to_dict()

    results["trend"] = run_trend_heatmap(cfg.trend).to_dict()

    return JSONResponse(content=results)
