# 🌊 Saltwater Intrusion Mapping Pipeline

> **An end-to-end, configuration-driven workflow for satellite imagery analysis of saltwater intrusion and coastal land cover change.**

This repository integrates Google Earth Engine (GEE) data acquisition, supervised land cover segmentation (FarSeg), heuristic salinity estimation, and temporal trend analysis. It is designed for both local research and scalable cloud-based training on Google Vertex AI.

---

## 📍 Table of Contents
* [🚀 Quick Start](#-quick-start)
* [📂 Repository Structure](#-repository-structure)
* [🗺️ High-Level Capabilities](#️-high-level-capabilities)
* [🔀 Workflow Stages](#-workflow-stages)
* [🧠 Segmentation Models](#-segmentation-models)
* [🧂 Salinity and Water Trends](#-salinity-and-water-trends)
* [☁️ Cloud and Docker](#️-cloud-and-docker)
* [📖 License](#-license)

---

## 🚀 Quick Start

### 1. Environment Setup 🛠️
Clone the repository and install the package in editable mode.

```
git clone https://github.com/psazama/saltwater-intrusion-mapping.git
cd saltwater-intrusion-mapping

conda env create -f environment.yml
conda activate saltmapping

pip install -e .
```

> You must have a **Google Earth Engine** account authenticated on your system:  
```
earthengine authenticate
```

---

### 2. Download-Only Workflow ⬇️

Skip all modeling steps and just download imagery:

```
python examples/workflow_runner.py --config examples/quickstart_download_only.toml
```

---

### 3. Training + Inference Example 🏋️

To train a FarSeg model on CDL labels and then run inference:

```
python examples/workflow_runner.py --config examples/quickstart_train.toml
```

This workflow will:

* Download imagery and CDL labels for training/validation
* Align labels to imagery mosaics
* Train a FarSeg segmentation model
* Run inference on trained model outputs

---

### 4. Run Inference with a Pre-trained Model Example 🏃‍♂️

The pipeline uses a Python workflow runner with TOML configuration files. Here’s a minimal inference example:

```
python examples/workflow_runner.py --config examples/quickstart_inference.toml
```

This workflow will:

* Download and mosaic imagery for the specified site
* Run FarSeg inference
* Save georeferenced prediction rasters and optional PNG previews

---

## 📂 Repository Structure

```
.
├── swmaps/                   # Main package source
│   ├── core/                 # GEE logic, satellites (L5, S2, L7), & spectral indices
│   ├── datasets/             # Data loaders for CDL, NLCD, and Salinity truth
│   ├── models/               # FarSeg, SAM, training logic, and inference
│   └── pipeline/             # Workflow stages (Trend, Masks, Salinity, Download)
├── config/                   # GeoJSON and GPKG region definitions
├── deploy/                   # Vertex AI deployment & hyperparameter search scripts
├── docs/                     # HTML documentation and API references
├── examples/                 # TOML configs and workflow runner entry points
├── notebooks/                # Visualization and output exploration tools
├── tests/                    # Pytest suite
└── pyproject.toml            # Build system and dependencies
```

---

## 🗺️ High-Level Capabilities

| Feature | Description |
| :--- | :--- |
| **🛰️ Multi-Sensor** | Native support for Landsat 5, 7, 8 and Sentinel-2 |
| **🧠 Deep Learning** | FarSeg architecture for semantic segmentation of coastal land cover |
| **🧂 Salinity** | Heuristic spectral classification using SWIR and turbidity proxies |
| **🌊 Water Masking** | NDWI-based water extent and temporal trend heatmaps |
| **☁️ Scaling** | One-command deployment to Vertex AI for A100 GPU training |

---

## 🔀 Workflow Stages

1. **📥 Download:** Imagery and auxiliary labels (CDL/NLCD) via GEE  
2. **🧩 Alignment:** Automatic reprojection and resampling of labels to imagery  
3. **🎓 Training:** Supervised segmentation model training (FarSeg)  
4. **🎯 Inference:** Batch processing of GeoTIFFs with spatial metadata preservation  
5. **🧪 Salinity:** Rule-based classification using spectral thresholds  
6. **📊 Analysis:** Temporal aggregation of water masks to generate trend heatmaps  

---

## 🧠 Segmentation Models

### Supported Architectures
* **FarSeg (Foreground-Aware Segmentation):** Optimized for sparse foregrounds like saltwater-affected areas  
* **SAM (Segment Anything Model):** Experimental zero-shot segmentation  

### Training Features
* Multi-site training with optional validation  
* Configurable batch size, learning rate, epochs, and loss function (Dice, Cross-Entropy, Focal)  
* Checkpointing of best-performing models  
* Learning rate scheduling and early stopping  

---

## 🧂 Salinity and Water Trends

### Heuristic Salinity Estimation

| Feature | Purpose | Bands (Sentinel-2) |
|:---|:---|:---|
| NDWI / MNDWI | Water detection | B3, B8, B11 |
| Turbidity Index | Suspended sediment proxy | B4/B3 |
| Salinity Proxy | Salt crust reflectance | B11 + B12 |
| Veg Mask | Plant stress indicator | NDVI |

**Python Usage Example:**

```python
from swmaps.core.salinity_tools import estimate_salinity_level

result = estimate_salinity_level(blue, green, red, nir, swir1, swir2)
# Returns {"class_map": "fresh/brackish/saline", "score": 0.0-1.0}
```

### Water Trend Analysis

* Inputs: multi-temporal stack of mosaics  
* Logic: pixel-wise linear regression on water frequency  

```python
from swmaps.core.water_trend import pixel_trend, plot_trend_heatmap

slope, pval = pixel_trend(wet_year_stack)
plot_trend_heatmap(slope, (pval < 0.05))
```

---

## ☁️ Cloud and Docker

### Vertex AI Deployment
* Single GPU training on A100  
* Hyperparameter search (loss functions, learning rates)  
* Spot instance scheduling and environment setup

```
./deploy/deploy_training.sh -p <your-gcp-project-id>
```

```
./deploy/deploy_hyperparameter_search.sh -p <your-gcp-project-id>
```

---

## 📖 License

This project is licensed under the **MIT License**.


