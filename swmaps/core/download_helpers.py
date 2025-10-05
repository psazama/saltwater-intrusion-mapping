"""Support routines for downloading and storing ancillary raster products."""

import io
import logging
import zipfile
from pathlib import Path
from typing import Sequence

import requests


def _region_tag(bounds: Sequence[float]) -> str:
    """Return a filesystem-friendly tag for the provided bounds.

    Args:
        bounds (Sequence[float]): Geographic bounds ordered as
            ``(minx, miny, maxx, maxy)`` in degrees.

    Returns:
        str: A sanitized string suitable for use in file names.
    """

    return "_".join(
        f"{coord:.6f}".replace("-", "m").replace(".", "p") for coord in bounds
    )


def _save_response_to_raster(content: bytes, destination: Path) -> Path:
    """Save a WCS/WMS response to a raster file, unzipping when necessary.

    Args:
        content (bytes): Raw response payload returned by the service.
        destination (Path): Target path where the GeoTIFF should be written.

    Returns:
        Path: The path of the written GeoTIFF, matching ``destination``.

    Raises:
        ValueError: If a ZIP archive is provided without any GeoTIFF members.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Many services return zipped GeoTIFFs. Detect and extract them if present.
    if content[:2] == b"PK":  # ZIP file signature
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_names = [
                name
                for name in zf.namelist()
                if name.lower().endswith((".tif", ".tiff"))
            ]
            if not tif_names:
                raise ValueError("ZIP archive did not contain any GeoTIFF files.")
            # If multiple GeoTIFFs are present, take the first one but warn the user.
            if len(tif_names) > 1:
                logging.warning(
                    "Multiple GeoTIFFs found in archive; defaulting to the first entry '%s'.",
                    tif_names[0],
                )
            with zf.open(tif_names[0]) as tif:
                destination.write_bytes(tif.read())
    else:
        destination.write_bytes(content)

    return destination


def _request_raster_with_format_fallback(
    url: str,
    base_params: dict[str, object],
    format_preferences: Sequence[str],
    timeout: int = 300,
) -> tuple[requests.Response, str]:
    """Request a raster preferring formats earlier in ``format_preferences``.

    Parameters
    ----------
    url : str
        Endpoint to query.
    base_params : dict[str, object]
        Parameters to include with each request *except* ``format``.
    format_preferences : Sequence[str]
        Ordered sequence of format strings to try. The first successful
        response is returned. All remaining formats serve as fallbacks.
    timeout : int, default=300
        Request timeout in seconds.

    Returns
    -------
    tuple[requests.Response, str]
        The successful response object and the format string that produced it.

    Raises
    ------
    requests.HTTPError
        If none of the formats succeed.
    """

    last_error: requests.RequestException | None = None

    for fmt in format_preferences:
        params = dict(base_params)
        params["format"] = fmt
        logging.debug("Attempting request to %s with format '%s'", url, fmt)
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - network error handling
            logging.debug(
                "Request to %s with format '%s' failed: %s",
                url,
                fmt,
                exc,
            )
            last_error = exc
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if "xml" in content_type and "tiff" not in content_type:
            # Services sometimes return XML exception reports with HTTP 200.
            logging.debug(
                "Request to %s with format '%s' returned XML content; trying fallback format.",
                url,
                fmt,
            )
            last_error = requests.HTTPError(
                "Service returned XML response instead of raster.", response=response
            )
            continue

        logging.info("Successfully retrieved raster using format '%s'.", fmt)
        return response, fmt

    if last_error is not None:
        raise last_error

    raise requests.HTTPError(
        "All requested formats failed but no error response was captured."
    )
