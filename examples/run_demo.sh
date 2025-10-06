#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------
# End-to-end driver for the demo workflow
# --------------------------------------------------------------------

# Load config (defaults to config/demo.conf)
CONFIG_FILE="${1:-config/demo.conf}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
source "$CONFIG_FILE"

echo "=== Step 0: Create coastal polygon ==="
python examples/coastal_poly.py --use-bbox

echo "=== Step 1: Download imagery ==="
python examples/download_data.py --multithreaded --inline-mask --max-items 3

echo "=== Step 2: Generate NDWI water masks ==="
python examples/generate_masks.py --center-size 64

echo "=== Step 3: Compute water trend heatmap ==="
python examples/trend_heatmap.py

echo "=== Step 4: Run salinity pipeline (optional) ==="
if [[ -n "$TRUTH_DIR" || -n "$TRUTH_FILE" ]]; then
    python examples/salinity_pipeline.py \
        ${TRUTH_DIR:+--truth-dir "$TRUTH_DIR"} \
        ${TRUTH_FILE:+--truth-file "$TRUTH_FILE"}
else
    echo "Skipping salinity pipeline (no truth data provided)"
fi

echo "=== Step 5: Fetch NLCD and CDL overlays ==="
python examples/fetch_overlays.py --region "$REGION" --year $YEAR

echo "=== Step 6: Estimate salinity from Landsat mosaic (demo) ==="
# You can adjust this to point at a specific mosaic file created earlier
MOSAIC_FILE="swmaps/data/examples/somerset_landsat/landsat5_somerset_1990-07-01_1990-07-31.tif"
if [[ -f "$MOSAIC_FILE" ]]; then
    python examples/landsat_salinity.py --mosaic "$MOSAIC_FILE" --water-threshold $WATER_THRESHOLD
else
    echo "No Landsat mosaic found at $MOSAIC_FILE, skipping Landsat salinity demo"
fi

echo "✅ Somerset demo pipeline completed."

