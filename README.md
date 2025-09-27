# 🌊 Saltwater Intrusion Detection from Satellite Imagery

This project detects and visualizes saltwater intrusion in coastal agricultural areas using Sentinel-2, Landsat-5, Landsat-7, and other publicly available remote sensing datasets. It leverages patchwise querying, in-memory mosaicking, and multi-band analysis for scalable processing of large geospatial regions. The NDWI (Normalized Difference Water Index) and other spectral indices are applied to identify water-inundated zones over time.

-----

## 🚀 Features

  - Patch-based tiling of bounding boxes for scalable downloads and processing
  - STAC-based querying of Sentinel-2 and Landsat-5/7 imagery from AWS Open Data
  - Cloud-aware download skipping based on NaN proportion checks
  - In-memory mosaicking of image patches (compressed GeoTIFF output)
  - NDWI and custom index calculation to identify surface water and salinity indicators
  - Heuristic salinity classification combining NDWI/MNDWI, turbidity, chlorophyll, and SWIR proxies
  - Support for multiband TIFF reading and georeferenced patch extraction
  - Modular functions for salinity feature engineering and truth data extraction
  - Tools to track downloaded scenes and support fault-tolerant re-runs
  - Visualization support with Folium and Matplotlib

-----

## 🗺️ Use Cases

  - Monitoring farmland degradation due to saltwater intrusion
  - Surface water change detection over time
  - Scalable pre-processing for Earth observation ML workflows

-----

## 🗃️ Repository Layout

```
saltwater-intrusion-mapping/
├── config/ # study area & date ranges
├── swmaps/ # Python package
│ ├── core/ # download_tools, salinity_tools, …
│ ├── pipelines/ # Dagster assets/resources
│ └── data/ # generated rasters (git-ignored)
├── notebooks/ # experiments / visual demos
├── pipeline_runner.py # legacy CLI (still works)
├── dagster.yaml # local dev instance (optional)
├── dagster_gke.yaml # run launcher for GKE
├── workspace.yaml # loads swmaps.defs (from __init__.py)
├── workspace_gke.yaml # loads swmaps.gke_defs
└── pyproject.toml
```

-----

## ⚙️ `pipeline_runner.py` CLI Tool 

The `pipeline_runner.py` script is a command-line interface (CLI) that orchestrates the project's workflow. It allows you to run specific steps of the saltwater intrusion detection pipeline, from data download to trend analysis.

### How to Use

You must specify the desired processing step using the `--step` argument. The steps are numbered from 0 to 4.

**Syntax:**

```bash
python pipeline_runner.py --step <number> [options]
```

### Command-line Arguments

Here is a breakdown of the available arguments:

  * `--step` (required):

      * `0`: **Creates a coastal polygon** for the analysis area.
      * `1`: **Downloads satellite imagery** mosaics based on a date range.
      * `2`: **Generates water masks** from the downloaded mosaics.
      * `3`: **Performs water trend analysis** and generates a heatmap.
      * `4`: **Runs the salinity data pipeline** to prepare data for modeling.

  * `--salinity_truth_directory`: Path to a directory containing `.nc` files to build a ground truth salinity dataset.

  * `--salinity_truth_file`: Path to the CSV file where the salinity ground truth data is stored or loaded from. Defaults to `data/salinity_labels/codc_salinity_profiles.csv`.

  * `--inline_mask`: If this flag is set, a water mask is created immediately after a mosaic is downloaded, and the original mosaic is deleted to save disk space.

  * `--n_workers`: The number of parallel workers to use for imagery processing. Defaults to half of the available CPU cores.

  * `--bbox`: Use the full bounding box from the GeoJSON file instead of the coastal band.

  * `--max_items`: The maximum number of images to download for each region. Defaults to 1.

  * `--multithreaded`: Use a multi-threaded version of the download function for faster processing.

  * `--center_size`: Specifies the size of the center of the image to check for NDWI calculation.

### Examples

**1. Create a coastal polygon:**

```bash
python pipeline_runner.py --step 0
```

**2. Download satellite imagery for test geojson bounding box:**

```bash
python pipeline_runner.py --step 1 --bbox
```

**3. Generate water masks and delete original mosaics:**

```bash
python pipeline_runner.py --step 2 --inline_mask
```

**4. Analyze water trends and plot a heatmap:**

```bash
python pipeline_runner.py --step 3
```

**5. Run the full salinity pipeline:**

```bash
python pipeline_runner.py --step 4 --salinity_truth_directory /path/to/salinity/data
```

-----

## 🧂 Water Salinity Estimation

| **Feature** | **Sentinel-2 Bands** | **Purpose** |
|---|---|---|
| NDWI / MNDWI | B3 (green), B8 (NIR), B11 (SWIR) | Water detection |
| Turbidity Index | B4 (red) / B3 (green), B4 / B8 | Suspended sediment proxy |
| Chlorophyll Index | (B5 − B4)/(B5 + B4), or B3/B2 | Low chlorophyll can indicate salinity |
| Salinity Proxy Index (custom) | B11 + B12 (SWIR) | High reflectance in saline water/salt crusts |
| NDTI (Normalized Difference Turbidity Index) | (B3 − B2)/(B3 + B2) | Surface turbidity |
| Salinity-sensitive Vegetation Mask | NDVI around water | Nearby plant stress as salinity indicator |

### Estimating salinity classes in code

The helper `estimate_salinity_level` in `swmaps.core.salinity_tools` combines the proxies above to
return a per-pixel salinity score and qualitative class (fresh, brackish, saline). Provide the
individual band arrays (either in raw Sentinel-2 scale 0–10,000 or already scaled reflectances) and
the function handles the rest:

```python
from swmaps.core.salinity_tools import estimate_salinity_level

result = estimate_salinity_level(blue, green, red, nir, swir1, swir2)
class_map = result["class_map"]  # string labels per pixel
salinity_score = result["score"]  # 0–1 heuristic intensity (NaN outside water)
```

Tune the optional thresholds (e.g., `water_threshold`, `salinity_proxy_threshold`) if you have
region-specific calibration data.

-----

## 🌡️ Water Trend Analysis

Use `swmaps.core.water_trend` to model how long each pixel stays water-covered and how that changes over time.

```python
from swmaps.core.water_trend import (
    load_wet_year,
    pixel_trend,
    plot_trend_heatmap,
    save_trend_results,
)

wet_year = load_wet_year("masks/*.tif")
slope, pval = pixel_trend(wet_year)
signif = pval < 0.05
plot_trend_heatmap(slope, signif, title="Trend in % wet months per year")
# Save arrays to GeoTIFF and NumPy for later inspection
save_trend_results(slope, pval, "water_trend")
```

-----

## ⚡ Quick Start Orchestrator (local)

```bash
# 1 · Install (dev mode)
git clone https://github.com/<you>/saltwater-intrusion-mapping.git
cd saltwater-intrusion-mapping
conda env create -f swmaps/core/environment.yml    # or: pip install -r requirements.txt
pip install -e .

# 2 · Launch Dagster UI (with queued run-coordinator)
export DAGSTER_HOME="$(pwd)/.dagster_home"
dagster dev -w workspace.yaml    # → http://localhost:3000
# (the dagster.yaml file is optional; Dagster will fall back to defaults)

# 3 · Materialise water masks
#     (UI → Assets → masks_by_range → Launch backfill)
```

> Smoketest: run without Dagster
> `python pipeline_runner.py --step 0 --inline_mask`

-----

## 🚀 Running on GKE

The repository includes a `dagster_gke.yaml` configuration that launches each
Dagster run as a Kubernetes job. After building a container image for the
`swmaps` package, point your deployment at this config file:

```bash
kubectl create namespace dagster
export DAGSTER_HOME=/opt/dagster
dagster api grpc -m swmaps.gke_defs &
dagster-webserver -y dagster_gke.yaml -w workspace_gke.yaml
```

-----

## 📖 License

MIT License
