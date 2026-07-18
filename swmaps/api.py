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
import os
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from swmaps import __version__
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
TITILER_URL = os.environ.get("TITILER_URL", "http://localhost:8001")

# Shared secret for the pipeline-triggering endpoints. When unset, the
# endpoints stay open for local development but a warning is logged.
API_KEY = os.environ.get("SWMAPS_API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency guarding the pipeline endpoints.

    Compares the ``X-API-Key`` request header against the ``SWMAPS_API_KEY``
    environment variable using a constant-time comparison.
    """
    if API_KEY is None:
        logger.warning(
            "SWMAPS_API_KEY is not set - pipeline endpoints are UNAUTHENTICATED. "
            "Set it in any deployment reachable by untrusted clients."
        )
        return
    if not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Background job runner
#
# Pipeline steps can take minutes to hours; running them inside the request
# handler blocks a worker and times out clients. Jobs are submitted to a small
# thread pool and polled via GET /jobs/{job_id}. The registry is in-memory,
# so job state is lost on restart - completed runs are still recorded in the
# processed_products table where applicable.
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("SWMAPS_PIPELINE_WORKERS", "2"))
)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _submit_job(task: str, fn, /, *args, **kwargs) -> JSONResponse:
    """Run *fn* in the background and return a 202 with a pollable job id."""
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "task": task,
            "status": "running",
            "result": None,
            "error": None,
        }

    def _run() -> None:
        try:
            result = fn(*args, **kwargs)
            payload = result if isinstance(result, dict) else result.to_dict()
            with _jobs_lock:
                _jobs[job_id].update(status="complete", result=payload)
        except Exception as exc:
            logger.exception("Background job %s (%s) failed", job_id, task)
            with _jobs_lock:
                _jobs[job_id].update(status="failed", error=str(exc))

    _executor.submit(_run)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "task": task,
            "status": "running",
            "poll": f"/jobs/{job_id}",
        },
    )


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
    version=__version__,
    lifespan=lifespan,
)

# Serve React build
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the React frontend."""
    index = static_dir / "index.html"
    if not index.exists():
        return JSONResponse(
            {"message": "Frontend not built. Run npm run build in frontend/"}
        )
    return FileResponse(index)


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
        ) from exc


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
            ) from None

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
        examples=["-76.0,38.0,-75.0,39.0"],
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


@app.post(
    "/run/download", tags=["pipeline"], dependencies=[Depends(require_api_key)]
)
def trigger_download(cfg: DownloadConfig) -> JSONResponse:
    """Trigger the imagery download pipeline step in the background.

    Posts a :class:`~swmaps.schema.DownloadConfig`; returns ``202`` with a
    job id to poll at ``/jobs/{job_id}``.
    """
    return _submit_job("download", run_download, cfg)


@app.post("/run/masks", tags=["pipeline"], dependencies=[Depends(require_api_key)])
def trigger_masks(
    input_dir: str = Query(..., description="Directory of mosaics to process"),
) -> JSONResponse:
    """Trigger water mask generation in the background.

    Returns ``202`` with a job id to poll at ``/jobs/{job_id}``.
    """

    def _job() -> object:
        with get_connection() as conn:
            return run_water_masks(Path(input_dir), conn=conn)

    return _submit_job("water_masks", _job)


@app.post(
    "/run/salinity", tags=["pipeline"], dependencies=[Depends(require_api_key)]
)
def trigger_salinity(cfg: SalinityConfig) -> JSONResponse:
    """Trigger the salinity ground-truth pipeline in the background.

    Posts a :class:`~swmaps.schema.SalinityConfig`; returns ``202`` with a
    job id to poll at ``/jobs/{job_id}``.
    """
    return _submit_job("salinity_pipeline", run_salinity_pipeline, cfg)


@app.post(
    "/run/salinity/classify",
    tags=["pipeline"],
    dependencies=[Depends(require_api_key)],
)
def trigger_salinity_classify(
    cfg: SalinityConfig,
    input_dir: str = Query(..., description="Directory of mosaics to classify"),
) -> JSONResponse:
    """Trigger per-mosaic salinity classification in the background.

    Returns ``202`` with a job id to poll at ``/jobs/{job_id}``.
    """

    def _job() -> object:
        with get_connection() as conn:
            return run_salinity_classification(cfg, Path(input_dir), conn=conn)

    return _submit_job("salinity_classification", _job)


@app.post("/run/trend", tags=["pipeline"], dependencies=[Depends(require_api_key)])
def trigger_trend(cfg: TrendConfig) -> JSONResponse:
    """Trigger the water-trend heatmap pipeline step in the background.

    Posts a :class:`~swmaps.schema.TrendConfig`; returns ``202`` with a
    job id to poll at ``/jobs/{job_id}``.
    """
    return _submit_job("trend", run_trend_heatmap, cfg)


@app.post(
    "/run/workflow", tags=["pipeline"], dependencies=[Depends(require_api_key)]
)
def trigger_workflow(cfg: WorkflowConfig) -> JSONResponse:
    """Trigger the full end-to-end workflow in the background.

    Returns ``202`` with a job id; the job result is a dict of
    :class:`~swmaps.schema.PipelineResult` payloads keyed by stage name.
    """

    def _job() -> dict:
        results = {}
        results["download"] = run_download(cfg.download).to_dict()
        results["salinity_pipeline"] = run_salinity_pipeline(cfg.salinity).to_dict()

        with get_connection() as conn:
            results["salinity_classification"] = run_salinity_classification(
                cfg.salinity, Path(cfg.download.out_dir or "data/outputs"), conn=conn
            ).to_dict()
            results["water_masks"] = run_water_masks(
                Path(cfg.download.out_dir or "data/outputs"),
                conn=conn,
            ).to_dict()

        results["trend"] = run_trend_heatmap(cfg.trend).to_dict()
        return results

    return _submit_job("workflow", _job)


@app.get("/jobs/{job_id}", tags=["pipeline"])
def get_job(job_id: str) -> dict:
    """Poll the status of a background pipeline job.

    Args:
        job_id: Identifier returned by a ``POST /run/*`` endpoint.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job = job.copy()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/preview", tags=["scenes"])
def preview_product(path: str = Query(..., description="Local file path to preview")):
    """Serve a product PNG for preview in the science viewer.

    Paths are resolved against the configured data root, and the resolved
    path must remain inside it - absolute paths and ``..`` segments cannot
    be used to read arbitrary files on the host.
    """
    from swmaps.config import settings

    base = settings.data_root.resolve()
    requested = Path(path)
    file_path = (
        requested.resolve() if requested.is_absolute() else (base / requested).resolve()
    )

    # Containment check: reject anything that escapes the data root.
    try:
        file_path.relative_to(base)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Path is outside the configured data root.",
        ) from None

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # For TIF files, look for a companion PNG
    if file_path.suffix.lower() == ".tif":
        png_path = file_path.with_suffix(".png")
        if png_path.exists():
            file_path = png_path
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No PNG preview found for {path}. Run the pipeline with save_png=true.",
            )

    if file_path.suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Only PNG files can be previewed")

    return FileResponse(file_path, media_type="image/png")


@app.get("/sensors", tags=["status"])
def list_sensors() -> dict:
    """List all registered satellite mission slugs.

    Returns:
        dict: Mission slugs available for filtering.
    """
    from swmaps.core.missions import _MISSION_REGISTRY

    return {"sensors": sorted(_MISSION_REGISTRY.keys())}


@app.get("/config", tags=["status"])
def get_config() -> dict:
    """Return runtime configuration for the frontend.

    Returns:
        dict: Frontend config including the TiTiler base URL.
    """
    return {
        "titiler_url": os.environ.get("TITILER_URL", "http://localhost:8001"),
    }


# ---------------------------------------------------------------------------
# Titiler tile endpoints
# ---------------------------------------------------------------------------


@app.get("/tiles/{path:path}", tags=["tiles"])
async def proxy_tiles(path: str, request: Request):
    """Proxy tile requests to TiTiler."""
    params = dict(request.query_params)
    url = f"{TITILER_URL}/{path}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "image/png"),
    )
