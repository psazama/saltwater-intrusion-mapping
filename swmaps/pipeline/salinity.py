"""Salinity pipeline utilities for aligning truth data and imagery products."""

from pathlib import Path

from swmaps.config import data_path
from swmaps.core.missions import get_mission
from swmaps.core.salinity.utils import (
    build_salinity_truth,
    download_salinity_datasets,
    extract_salinity_features_from_mosaic,
    load_salinity_truth,
)
from swmaps.core.satellite_query import (
    download_matching_images,
    find_satellite_coverage,
)


def salinity_pipeline(truth_download_list=None, truth_dir=None, truth_file=None):
    """Run salinity ground-truth processing and feature extraction.

    Args:
        truth_dir (str | Path | None): Directory containing raw CODC NetCDF
            profiles. When provided, profiles are ingested prior to matching.
        truth_file (str | Path | None): Existing truth CSV to use or create.

    Returns:
        None: Artifacts and CSV outputs are written to ``data/``.
    """
    if truth_download_list:
        if truth_dir:
            download_salinity_datasets(truth_download_list, truth_dir)
        else:
            download_salinity_datasets(truth_download_list)

    if truth_file is None:
        truth_file = str(data_path("salinity_labels", "codc_salinity_profiles.csv"))

    if truth_dir:
        codc_files = sorted(Path(truth_dir).glob("*.nc"))
        build_salinity_truth(codc_files, truth_file)

    y = load_salinity_truth(truth_file)
    overlapped = find_satellite_coverage(y)
    overlapped = overlapped[overlapped["covered_by"].apply(len) > 0]

    matches = download_matching_images(overlapped)
    csv_path = data_path("ground_truth_matches.csv")
    matches.to_csv(csv_path, index=False)
    print(f"Downloaded matches listed in {csv_path}")

    sample = matches.iloc[0]
    mission = get_mission(sample["covered_by"][0])
    mosaic = sample["downloaded_files"][0]
    base = Path(mosaic).with_suffix("")
    extract_salinity_features_from_mosaic(
        mosaic, mission["band_index"], f"{base}_features.tif", f"{base}_mask.tif"
    )
