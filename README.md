# 🌊 Saltwater Intrusion Detection from Satellite Imagery

This project provides a modular pipeline for detecting and visualizing saltwater intrusion in coastal agricultural regions. It combines remote sensing data, geospatial processing, and machine‑learning feature engineering to track changes in surface water and salinity over time.

The core library retrieves imagery from multiple satellite missions, including Sentinel‑2 and the Landsat series, assembles mosaics for areas of interest, computes spectral indices, and applies heuristic rules to derive salinity indicators. Additional modules provide tools for trend analysis, salinity truth matching, and optional integration with orchestration frameworks.

-----

## 🚀 Features

 - Remote‑sensing imagery retrieval - loads optical data from multiple missions and supports bounding‑box queries for scalable processing.
 - Patch‑based mosaic construction - downloads small tiles around each observation point and stitches them into multi‑band GeoTIFF mosaics
 - Index computation - calculates NDWI/MNDWI and other water proxies, turbidity and chlorophyll ratios, and salinity‑sensitive SWIR indices
 - Heuristic salinity classification - combines multiple indices into per‑pixel scores and qualitative salinity classes (fresh, brackish, saline)
 - Water mask and trend analysis - generates binary masks and models pixel‑level inundation frequency over time to highlight long‑term changes
 - Modular Python API - separates reusable core functions (swmaps.core) from higher‑level orchestration (swmaps.pipeline) so you can integrate specific pieces into your own workflows
 - Config‑driven workflows - run full workflows by supplying a simple TOML file that defines dates, location, mission, and other parameters.
 - Optional advanced orchestration - includes Dagster definitions and a Kubernetes configuration for scaling out jobs in the cloud.

-----

## 🗺️ Use Cases

- Monitoring farmland degradation caused by saltwater intrusion and sea‑level rise.
- Detecting changes in surface water and flooding over seasonal or multi‑year periods.
- Pre‑processing satellite imagery for downstream machine‑learning tasks, such as classification or regression models in Earth observation.

-----

## 🗃️ Repository Layout

```
saltwater-intrusion-mapping/
├── config/            - definitions of study areas, date ranges and example config templates
├── swmaps/            - Python package with core utilities (gee_query, mission metadata, mosaic, salinity tools, water trends) and pipeline helpers
├── examples/          - command‑line scripts and TOML files to run the pipeline (`workflow_runner.py`, example configs)
├── notebooks/         - Jupyter notebooks for experiments and visual demos
├── docs/              - HTML documentation generated via Sphinx
├── tests/             - unit tests
├── pyproject.toml     - project metadata and dependencies
└── README.md          - project overview
```

-----

## ⚙️ Running the Pipeline

1. Install - clone the repository, create a clean environment, and install the package. You’ll also need to install the Earth Engine API if it’s not already included.

```bash
git clone https://github.com/psazama/saltwater-intrusion-mapping.git
cd saltwater-intrusion-mapping
pip install -e .
```

2. Authenticate - follow the Earth Engine authentication flow to link your Google account and project, then initialise the API. Alternatively, set the EARTHENGINE_PROJECT environment variable and call ee.Initialize()
3. Create a configuration file - copy one of the TOML templates under examples/ and customise start/end dates, latitude/longitude, mission, buffer size, cloud filter and other parameters
```bash
python examples/workflow_runner.py --config examples/choptank.toml
```
4. Run the workflow - use the provided workflow_runner.py script with your configuration file. The runner will optionally build a coastal AOI, download mosaics, and run the salinity pipeline, saving results to your chosen output directory
5. Salinity analysis - if you have ground‑truth salinity data, call the salinity_pipeline.py script to extract features and match them to your truth data. You can also import and use functions like estimate_salinity_level from swmaps.core.salinity_tools directly in your own code.

-----

## 🧂 Water Salinity Estimation

Provides routines to compute multiple indices and combine them into a salinity score and classification. See the docstring of estimate_salinity_level in swmaps/core/salinity_tools.py for details

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
individual band arrays (either in raw Sentinel-2 scale 0-10,000 or already scaled reflectances) and
the function handles the rest:

```python
from swmaps.core.salinity_tools import estimate_salinity_level

result = estimate_salinity_level(blue, green, red, nir, swir1, swir2)
class_map = result["class_map"]  # string labels per pixel
salinity_score = result["score"]  # 0-1 heuristic intensity (NaN outside water)
```

Tune the optional thresholds (e.g., `water_threshold`, `salinity_proxy_threshold`) if you have
region-specific calibration data.

-----

## 🌡️ Water Trend Analysis

Includes utilities to assemble yearly wet masks, run pixel‑wise trend regressions, and plot heatmaps of inundation frequency

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

## 📖 License

MIT License
