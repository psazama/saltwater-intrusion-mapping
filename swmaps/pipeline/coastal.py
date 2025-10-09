"""Pipeline helpers for preparing coastal region geometry inputs."""

from pathlib import Path

from swmaps.core.coastline import create_coastal_poly


def create_coastal(
    use_bbox: bool = False, output_root: str | Path | None = None
) -> None:
    """Create the study coastal polygon or bounding-box region.

    Args:
        use_bbox (bool): If ``True``, load the configured bounding box instead
            of generating the buffered coastal polygon.
        output_root (str | Path | None): Directory where the generated
            ``coastal_band.gpkg`` should be written. When ``None``, defaults to
            the project ``config/`` directory.

    Returns:
        None: Output files are written under ``output_root`` or ``config/``.
    """
    config_dir = Path(__file__).resolve().parents[2] / "config"
    bbox_file = (
        config_dir / "somerset.geojson"
        if use_bbox
        else config_dir / "coastal_band.geojson"
    )

    destination_dir = Path(output_root) if output_root else config_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    out_file = destination_dir / "coastal_band.gpkg"

    print("Creating Coastal Polygon")
    create_coastal_poly(bbox_file, out_file=out_file)
