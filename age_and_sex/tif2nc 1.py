import os, re
from glob import glob
import rioxarray as rxr
import xarray as xr

sex_list = ['m', 'f', 't']
main_path = os.getcwd()

if os.path.exists("population_municipality.nc"):
    ds_muni = xr.open_dataset("population_municipality.nc")
if os.path.exists("population_non_municipality.nc"):
    ds_non_muni = xr.open_dataset("population_non_municipality.nc")
else:
    datasets_muni = []
    datasets_non_muni = []

    # Reference grid
    ref_file = glob(f'{main_path}\\grc_*.tif')[0]
    ref = rxr.open_rasterio(ref_file, masked=True).squeeze()

    for sex in sex_list:
        tif_files = glob(f'{main_path}\\grc_{sex}*.tif')
        for f in tif_files:
            # Extract year
            year_match = re.search(r"(\d{4})", f)
            year = int(year_match.group(1)) if year_match else None

            # Extract age group
            age_match = re.search(r"grc_[mft]_(\d{2})_", f)
            age_group = age_match.group(1) if age_match else "unknown"

            # Municipality flag
            is_muni = "municipality" in f

            # Open raster lazily
            da = rxr.open_rasterio(f, masked=True, chunks={'x':1024, 'y':1024}).squeeze()

            # Expand dimensions: time, sex, age_group
            da = da.expand_dims({
                "time": [year],
                "sex": [sex],
                "age_group": [age_group]
            })

            if is_muni:
                datasets_muni.append(da)
            else:
                datasets_non_muni.append(da)

    # Combine by coords
    ds_muni = xr.combine_by_coords(datasets_muni, combine_attrs="override")
    ds_muni = xr.Dataset({"population": ds_muni})
    ds_muni.chunk({'time': 1, 'sex': 1, 'age_group': 1}).to_netcdf("population_municipality.nc")

    ds_non_muni = xr.combine_by_coords(datasets_non_muni, combine_attrs="override")
    ds_non_muni = xr.Dataset({"population": ds_non_muni})
    ds_non_muni.chunk({'time': 1, 'sex': 1, 'age_group': 1}).to_netcdf("population_non_municipality.nc")

# Sum over spatial dimensions (x, y)
ds_muni_sum = ds_muni["population"].sum(dim=["x", "y", "age_group"])
ds_non_muni_sum = ds_non_muni["population"].sum(dim=["x", "y", "age_group"])

# Convert to DataFrame
df_muni = ds_muni_sum.to_dataframe().reset_index()
df_non_muni = ds_non_muni_sum.to_dataframe().reset_index()

# Pivot: columns = sex, index = time, age_group
df_muni_pivot = df_muni.pivot_table(index=["time"], columns="sex", values="population", fill_value=0)
df_non_muni_pivot = df_non_muni.pivot_table(index=["time"], columns="sex", values="population", fill_value=0)

print(df_muni_pivot)
print(df_non_muni_pivot)