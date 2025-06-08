from data_downloader import *
import geopandas as gpd
import json
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Saltwater Intrustion Detection Runner")
    parser.add_argument(
        "--step", type=int, default=0, required=False, choices=[0,1,2],
        help="Processing step to begin on (0 = data download, 1 = water mask creation, 2 = water map time comparison)"
    )
    args = parser.parse_args()

    # Load the GeoJSON
    gdf = gpd.read_file("easternshore.geojson")
    
    # Ensure it's in WGS84 (EPSG:4326) for STAC API compatibility
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox = gdf.total_bounds.tolist()
    sentinel_mission = get_mission("sentinel-2")
    landsat5_mission = get_mission("landsat-5")
    landsat5_mosaic_path = "data/landsat_eastern_shore.tif"
    sentinel2_mosaic_path = "data/sentinel_eastern_shore.tif"

    if args.step <= 0:            
        with open('date_range.json', 'r') as f:
            dates = json.load(f)
        
        for date in tqdm(dates['date_ranges'][-84::3]):
            try:
                mname = landsat5_mosaic_path[:-4] + "_" + date.replace("/","_") + ".tif"
                print(mname)
                query_satellite_items(mission="landsat-5", bbox=bbox, date_range=date, max_items=1, debug=False)
                create_mosaic_placeholder(
                    mosaic_path=mname,
                    bbox=reproject_bbox(bbox),
                    resolution=landsat5_mission["resolution"],
                    mission="landsat-5",
                    crs="EPSG:32618",
                    dtype="float32"
                )
                patchwise_query_download_mosaic(mname, 
                                                bbox, 
                                                "landsat-5", 
                                                landsat5_mission["resolution"]*1000, 
                                                landsat5_mission["resolution"], 
                                                landsat5_mission["bands"], 
                                                date, 
                                                None, 
                                                to_disk=False
                                               )
            except Exception as e:
                print(e)
            try:
                mname = sentinel2_mosaic_path[:-4] + "_" + date.replace("/","_") + ".tif"
                print(mname)
                query_satellite_items(mission="sentinel-2", bbox=bbox, date_range=date, max_items=1, debug=False)
                create_mosaic_placeholder(
                    mosaic_path=mname,
                    bbox=reproject_bbox(bbox),
                    resolution=sentinel_mission["resolution"],
                    mission="sentinel-2",
                    crs="EPSG:32618",
                    dtype="float32"
                )
                patchwise_query_download_mosaic(mname, 
                                                bbox, 
                                                "sentinel-2", 
                                                sentinel_mission["resolution"]*1000, 
                                                sentinel_mission["resolution"], 
                                                sentinel_mission["bands"], 
                                                date,  
                                                None, 
                                                to_disk=False
                                               )
            except Exception as e:
                print(e)
    if args.step <= 1:    
        pass
        # compute_ndwi(green_path, nir_path, out_path=None, display=False):


if __name__ == "__main__":
    main()
