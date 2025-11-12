import glob, re
import rioxarray as rxr
import xarray as xr
import pandas as pd
 
# Define variables
sex_list = ['m', 'f', 't']
 
# 1. Find all matching .tif files
main_path = 'C:/Users/xylop/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/age_and_sex/'
df_all_sexes = pd.DataFrame()
for sex in sex_list:
    tif_files = glob.glob(f'{main_path}grc_{sex}*v1_municipality_clipped.tif')
 
    results = []
 
    for f in tif_files:
        match_year = re.search(r"(\d{4})", f)
        if match_year:
            year = int(match_year.group(1))
        else:
            year = None
 
        # open raster
        da = rxr.open_rasterio(f, masked=True).squeeze()  # 2D array
        # sum all valid cells
        total_population = float(da.sum(skipna=True).values)
        # store result
        results.append({"year": year, "total_population": int(total_population)})
 
    # 3. Create dataframe
    df = pd.DataFrame(results).sort_values("year").set_index("year")
 
    annual = df.groupby(level=0)['total_population'].sum().sort_index()
    df_annual = pd.DataFrame(data={sex:annual.values}, index=annual.index)
    df_all_sexes = pd.concat([df_all_sexes, df_annual], axis=1)
df_all_sexes.to_csv('annual_sex_population_totals_municipality.csv', sep=',', index=True)
print(df_all_sexes)