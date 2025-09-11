# 🌊 Saltwater Intrusion Detection from Satellite Imagery

This project detects and visualizes saltwater intrusion in coastal agricultural areas using Sentinel-2, Landsat-5, Landsat-7, and other publicly available remote sensing datasets. It leverages patchwise querying, in-memory mosaicking, and multi-band analysis for scalable processing of large geospatial regions. The NDWI (Normalized Difference Water Index) and other spectral indices are applied to identify water-inundated zones over time.

## 🚀 Features
- Patch-based tiling of bounding boxes for scalable downloads and processing
- STAC-based querying of Sentinel-2 and Landsat-5/7 imagery from AWS Open Data
- Cloud-aware download skipping based on NaN proportion checks
- In-memory mosaicking of image patches (compressed GeoTIFF output)
- NDWI and custom index calculation to identify surface water and salinity indicators
- Support for multiband TIFF reading and georeferenced patch extraction
- Modular functions for salinity feature engineering and truth data extraction
- Tools to track downloaded scenes and support fault-tolerant re-runs
- Visualization support with Folium and Matplotlib

## 🗺️ Use Cases
- Monitoring farmland degradation due to saltwater intrusion
- Surface water change detection over time
- Scalable pre-processing for Earth observation ML workflows

## 🧂 Water Salinity Estimation
| **Feature**                      | **Sentinel-2 Bands**        | **Purpose**                                          |
|----------------------------------|------------------------------|------------------------------------------------------|
| NDWI / MNDWI                    | B3 (green), B8 (NIR), B11 (SWIR) | Water detection                                    |
| Turbidity Index                 | B4 (red) / B3 (green), B4 / B8 | Suspended sediment proxy                         |
| Chlorophyll Index              | (B5 − B4)/(B5 + B4), or B3/B2  | Low chlorophyll can indicate salinity              |
| Salinity Proxy Index (custom)  | B11 + B12 (SWIR)              | High reflectance in saline water/salt crusts       |
| NDTI (Normalized Difference Turbidity Index) | (B3 − B2)/(B3 + B2)   | Surface turbidity                                  |
| Salinity-sensitive Vegetation Mask | NDVI around water         | Nearby plant stress as salinity indicator          |

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


---

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

---

## ⚡ Quick Start (local)

```bash
# 1 · Install (dev mode)
git clone https://github.com/<you>/saltwater-intrusion-mapping.git
cd saltwater-intrusion-mapping
conda env create -f swmaps/core/environment.yml   # or: pip install -r requirements.txt
pip install -e .

# 2 · Launch Dagster UI (with queued run-coordinator)
export DAGSTER_HOME="$(pwd)/.dagster_home"
dagster dev -w workspace.yaml   # → http://localhost:3000
# (the dagster.yaml file is optional; Dagster will fall back to defaults)

# 3 · Materialise water masks
#     (UI → Assets → masks_by_range → Launch backfill)
```
> Smoketest: run without Dagster
> `python pipeline_runner.py --step 0 --inline_mask`

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

---

## 📖 License

MIT License

