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

## 🧂 Water Salinity Estimation
| **Feature**                      | **Sentinel-2 Bands**        | **Purpose**                                          |
|----------------------------------|------------------------------|------------------------------------------------------|
| NDWI / MNDWI                    | B3 (green), B8 (NIR), B11 (SWIR) | Water detection                                    |
| Turbidity Index                 | B4 (red) / B3 (green), B4 / B8 | Suspended sediment proxy                         |
| Chlorophyll Index              | (B5 − B4)/(B5 + B4), or B3/B2  | Low chlorophyll can indicate salinity              |
| Salinity Proxy Index (custom)  | B11 + B12 (SWIR)              | High reflectance in saline water/salt crusts       |
| NDTI (Normalized Difference Turbidity Index) | (B3 − B2)/(B3 + B2)   | Surface turbidity                                  |
| Salinity-sensitive Vegetation Mask | NDVI around water         | Nearby plant stress as salinity indicator          |


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