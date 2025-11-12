'''
Group Desctiption
---------------------
Group              | Age Range
Infants & children | 0–14
Young adults       | 15–29
Middle-aged adults | 30–49
Mature adults      | 50–64
Seniors            | 65+
--------------------
'''

import glob, re
import rioxarray as rxr
import xarray as xr
import pandas as pd
 
# Define variables
sex_list = ['m', 'f', 't']
age_groups = [["00","01","05",'10','15'], ["20","25","30"], ["35","40","45","50"],
              ["55","60","65"], ["70","75","80","85","90","95","100"]]
age_group_names = ["Infants_and_children", "Young_adults", "Middle_aged_adults",
                   "Mature_adults", "Seniors"]
 
# 1. Find all matching .tif files
main_path = 'C:/Users/xylop/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/age_and_sex/'
df_all_age_groups = pd.DataFrame()
for age_group, age_group_name in zip(age_groups, age_group_names):
    tif_files = [glob.glob(f'{main_path}grc_t_{age}*v1_clipped.tif') for age in age_group]
    tif_files = [item for sublist in tif_files for item in sublist]  # flatten list
 
    results = []
 
    for f in tif_files:
        match = re.search(r"(\d{4})", f)
        if match:
            year = int(match.group(1))
        else:
            year = None
 
        # open raster
        da = rxr.open_rasterio(f, masked=True).squeeze()  # 2D array
        # sum all valid cells
        group_population = float(da.sum(skipna=True).values)
        # store result
        results.append({"year": year, "group_population": int(group_population)})
 
    # 3. Create dataframe
    df = pd.DataFrame(results).sort_values("year").set_index("year")
 
    annual = df.groupby(level=0)['group_population'].sum().sort_index()
    df_annual = pd.DataFrame(data={age_group_name:annual.values}, index=annual.index)
    df_all_age_groups = pd.concat([df_all_age_groups, df_annual], axis=1)
df_all_age_groups.to_csv('age_groups_population.csv', sep=',', index=True)
print(df_all_age_groups)

