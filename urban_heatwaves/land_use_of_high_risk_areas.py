import rioxarray
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# --- inputs ---
risk_tif = "urban_heatwaves/heatwave_risk_map.tif"
landuse_fp = "urban_heatwaves/coast_line_2018_city_of_Kavala.gpkg"
landuse_field = "CODE_5_18"

# --- open raster with rioxarray ---
da = rioxarray.open_rasterio(risk_tif, masked=True).squeeze()  # 2D DataArray

# --- create mask for high-risk cells ---
mask = (da >= 15) & ~np.isnan(da)

# get the row/col indices of high-risk cells
rows, cols = np.where(mask.values)

# --- compute centroids using xarray’s coordinate system ---
# transform xarray coordinates (y, x)
x_coords = da.x.values[cols]
y_coords = da.y.values[rows]

# --- build GeoDataFrame of centroids ---
gdf_pts = gpd.GeoDataFrame(
    {"row": rows, "col": cols, "risk": da.values[rows, cols]},
    geometry=[Point(x, y) for x, y in zip(x_coords, y_coords)],
    crs=da.rio.crs
)

# --- read land-use layer ---
gdf_lu = gpd.read_file(landuse_fp)
if gdf_lu.crs != gdf_pts.crs:
    gdf_lu = gdf_lu.to_crs(gdf_pts.crs)

# --- spatial join to tag each centroid with land-use class ---
gdf_tagged = gpd.sjoin(
    gdf_pts,
    gdf_lu[[landuse_field, "geometry"]],
    how="left",
    predicate="within"
).drop(columns="index_right")

# --- summarize results ---
summary = gdf_tagged.groupby(landuse_field, dropna=False).size().sort_values(ascending=False)
print(summary)

# --- (optional) approximate area by cell count ---
px_area_m2 = abs(da.rio.resolution()[0] * da.rio.resolution()[1])
area_ha = (summary * px_area_m2) / 10_000
print("\nArea (ha) by land-use class (risk ≥ 15):")
print(area_ha.round(2))

# --- (optional) export points ---
gdf_tagged.to_file("urban_heatwaves/heatwave_risk_with_landuse.gpkg", driver="GPKG")

print()