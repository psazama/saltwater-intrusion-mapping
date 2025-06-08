# 🌊 Saltwater Intrusion Detection from Satellite Imagery

This project detects and visualizes saltwater intrusion in coastal agricultural areas using Sentinel-2 and Landsat-5 satellite imagery. It leverages patchwise querying and mosaicking for scalable processing of large geospatial regions and applies the NDWI (Normalized Difference Water Index) to identify water-inundated zones over time.
## 🚀 Features
- Patch-based tiling of bounding boxes for scalable downloads and processing
- STAC-based querying of Sentinel-2 and Landsat-5 imagery from AWS Open Data
- In-memory mosaicking of image patches (no intermediate disk writes required)
- NDWI calculation to identify surface water from Green/NIR bands
- Tools for windowed TIFF reading, patch extraction, and re-stitching
- Visualization support with Folium and Matplotlib

## 🗺️ Use Cases
- Monitoring farmland degradation due to saltwater intrusion
- Surface water change detection over time
- Scalable pre-processing for Earth observation ML workflows

## 📦 Dependencies
- rasterio
- geopandas
- numpy
- matplotlib
- pystac-client
- folium
- tqdm
- concurrent.futures
- pre-commit (optional, for clearing Jupyter cell outputs before commits)

## 📍 To Do / Next Steps
- Add temporal change detection (multi-date NDWI differencing)
- Classify affected zones for reporting
- Integrate elevation or soil salinity data
- Add web-based map viewer

## 📖 License

MIT License