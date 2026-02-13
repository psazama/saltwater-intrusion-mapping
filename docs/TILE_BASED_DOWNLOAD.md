# Tile-Based Download Strategy Implementation

## Overview

This document explains the tile-based download strategy implemented to fix the race condition in large Sentinel-2 image downloads.

## Problem

The previous implementation used asynchronous export to Google Drive for Sentinel-2 images larger than 30 MB. This caused a race condition where the code tried to read the file immediately after export, but the file didn't exist yet since the async task was still running in the background.

**Error:**
```
[GEE] Size exceeds 30 MB for Sentinel-2, exporting asynchronously: sentinel-2_20181122T155549_20181122T155550_T18SVH_multiband.tif
rasterio.errors.RasterioIOError: data/outputs/quickstart_download_only/1/sentinel-2/sentinel-2_20181122T155549_20181122T155550_T18SVH_multiband.tif: No such file or directory
```

## Solution

The implementation now uses a tile-based synchronous download strategy:

1. When a region exceeds the 30 MB threshold, break it into smaller sub-region tiles
2. Download each tile synchronously using GEE's `getDownloadURL()` 
3. Stitch all tiles together into a single output GeoTIFF using rasterio

## Implementation Details

### Helper Functions

Three new helper functions were added to `swmaps/core/satellite_query.py`:

#### `_get_tile_grid(bbox, num_tiles)`
Generates a grid of tile bounding boxes by dividing the region into equal parts.

**Parameters:**
- `bbox`: [minx, miny, maxx, maxy] - The bounding box to split
- `num_tiles`: Number of tiles per dimension (e.g., 2 means 2x2 = 4 tiles)

**Returns:**
- List of tile bounding boxes

**Example:**
```python
bbox = [0.0, 0.0, 10.0, 10.0]
tiles = _get_tile_grid(bbox, 2)
# Returns: [[0,0,5,5], [5,0,10,5], [0,5,5,10], [5,5,10,10]]
```

#### `_download_tile(image, bbox, band_list, scale, crs, tile_path)`
Downloads a single tile synchronously using GEE's `getDownloadURL()` method.

**Parameters:**
- `image`: The clipped and reprojected GEE image
- `bbox`: Tile bounding box [minx, miny, maxx, maxy]
- `band_list`: List of band names to download
- `scale`: Resolution in meters
- `crs`: Coordinate reference system
- `tile_path`: Output path for the tile

**Returns:**
- Path to the downloaded tile

#### `_stitch_tiles(tile_paths, output_path)`
Merges tiles into a single output GeoTIFF using rasterio's merge function.

**Parameters:**
- `tile_paths`: List of paths to tile GeoTIFFs
- `output_path`: Output path for the stitched GeoTIFF

**Returns:**
- Path to the stitched output file

### Updated Function

#### `download_gee_multiband()`
The main download function was updated to use the tile-based approach when the estimated size exceeds 30 MB for Sentinel-2 missions:

- For images 30-100 MB: Uses 2x2 grid (4 tiles)
- For images > 100 MB: Uses 3x3 grid (9 tiles)

## Benefits

1. **No Race Condition**: All files exist synchronously before being accessed
2. **Large Region Support**: Large regions are still handled (just split into manageable chunks)
3. **Identical Output**: The output is identical to what would have been exported asynchronously
4. **No External Dependencies**: No dependency on Google Drive or background task polling
5. **Better Error Handling**: Provides detailed error messages and continues with partial downloads
6. **Memory Efficient**: Uses context managers to avoid keeping all tiles open simultaneously

## Testing

Unit tests were added in `tests/test_satellite_query.py` to verify the tile grid generation works correctly:

- `test_get_tile_grid_2x2`: Verifies 2x2 tile grid generation
- `test_get_tile_grid_3x3`: Verifies 3x3 tile grid generation  
- `test_get_tile_grid_single`: Verifies single tile (1x1) grid
- `test_get_tile_grid_negative_coords`: Verifies handling of negative coordinates

All existing tests continue to pass, confirming backward compatibility.

## Usage

No changes are required in calling code. The function signature remains the same:

```python
from swmaps.core.satellite_query import download_gee_multiband

# Same API as before - tile-based download is automatic for large images
path = download_gee_multiband(
    image=ee_image,
    mission="sentinel-2",
    bands=band_dict,
    bbox=[minx, miny, maxx, maxy],
    out_dir=output_directory,
    scale=10
)
```

The function automatically detects large Sentinel-2 images and uses the tile-based approach transparently.
