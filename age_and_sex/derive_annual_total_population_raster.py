import rioxarray as rxr
import xarray as xr
import glob

# Read raster files and derive annual total
main_path = 'C:/Users/xylop/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/age_and_sex/'

years = [2015,2021,2029]

for year in years:
    tif_files = glob.glob(f'{main_path}grc_t_*{year}*v1_clipped.tif')

    # Open all rasters and store as a list of DataArrays
    rasters = [rxr.open_rasterio(f, masked=True) for f in tif_files]

    # Stack along a new dimension (e.g. "time")
    stacked = xr.concat(rasters, dim="band_sum")

    # Sum along that dimension (cell by cell)
    raster_sum = stacked.sum(dim="band_sum", skipna=True)

    # Save the result if needed
    raster_sum.rio.to_raster(f"{year}_sum_raster.tif")

raster_2015 = rxr.open_rasterio('2015_sum_raster.tif', masked=True)
raster_2029 = rxr.open_rasterio('2029_sum_raster.tif', masked=True)

raster_diff = raster_2015 - raster_2029
raster_diff.rio.to_raster('raster_difference_2015_2029.tif')

print()