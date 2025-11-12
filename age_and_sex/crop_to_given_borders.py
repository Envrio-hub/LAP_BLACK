import rioxarray as rxr
import zipfile
from pathlib import Path
from io import BytesIO
import geopandas as gpd
from shapely.ops import unary_union

TARGET_EPSG = 3035

sex_list = ['m', 'f', 't']

extract_dir = Path("temp_rasters")
zip_path = Path('C:/Users/xylop/Documents/github_repos/LAP_BLACK/zip_data')
vector_path = Path("C:/Users/xylop/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/municipality_of_kavala.shp")  # can be .shp/.geojson/.gpkg
output_dir = Path("C:/Users/xylop/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/age_and_sex/")

# Read vector layer once (in its native CRS)
gdf_orig = gpd.read_file(vector_path)

# Optional: dissolve multiple polygons into a single boundary
# (This avoids tiny slivers / gaps when clipping many rasters)
if len(gdf_orig) > 1:
    dissolved_geom = unary_union(gdf_orig.geometry)
    gdf_orig = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf_orig.crs)

# Fix potential geometry issues (self-intersections)
gdf_orig["geometry"] = gdf_orig.buffer(0)

# Collect rasters (clipped) for optional stacking at the end
clipped_rasters = []

# Loop ZIPs
for zip_file in sorted(zip_path.glob("*.zip")):
    print(f"/nProcessing ZIP: {zip_file.name}")
    with zipfile.ZipFile(zip_file, "r") as archive:

        for sex in sex_list:
        # List .tif entries
            tif_names = [n for n in archive.namelist() if n.lower().startswith(f"grc_{sex}") and n.lower().endswith(".tif")]
            if not tif_names:
                print("  (No .tif files found.)")
                continue

            for tif_name in tif_names:
                print(f"  Reading & clipping: {tif_name}")

                # Read raster in-memory
                with archive.open(tif_name) as f:
                    data = f.read()
                raster = rxr.open_rasterio(BytesIO(data), masked=True)

                # Skip rasters without CRS
                if raster.rio.crs is None:
                    print("    WARNING: Raster has no CRS. Skipping.")
                    continue

                # Reproject vector to the raster CRS
                gdf = gdf_orig.to_crs(raster.rio.crs)

                # Clip (drop=True trims the array to the geometry; invert=False keeps inside)
                try:
                    clipped = raster.rio.clip(
                        gdf.geometry, gdf.crs, drop=True, invert=False
                    )
                except Exception as e:
                    print(f"    ERROR clipping {tif_name}: {e}")
                    continue

                # Optional: carry a meaningful name as a coordinate
                # (helps later when stacking/identifying)
                clipped = clipped.assign_coords(
                    source_zip=zip_file.name,
                    source_path_in_zip=tif_name
                )

                # Save clipped raster to GeoTIFF
                out_name = f"{Path(tif_name).stem}_municipality_clipped.tif"
                out_path = output_dir / out_name
                clipped.rio.to_raster(out_path)
                print(f"    Saved: {out_path}")

            # Keep in memory for optional stacking
            # clipped_rasters.append(clipped)

