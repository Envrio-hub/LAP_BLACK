import glob, re
import rioxarray as rxr
import pandas as pd
 
# Age groups
age_categories = ["00","01","05","10", "15", "20", "25", "30", "35","40","45","50", "55","60","65", "70","75","80","85","90"]

# Years
years = [2015, 2016 , 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 , 2025, 2026, 2026, 2027, 2028, 2029]

# 1. Find all matching .tif files
main_path = 'C:/Users/xylop/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/age_and_sex/'


df_years = pd.DataFrame()
for year in years:
    df_ages = pd.DataFrame()
    for age_category in age_categories:
        tif_file = glob.glob(f'{main_path}grc_t_{age_category}_{year}*v1_clipped.tif')
    
        results = []
    
        # open raster
        da = rxr.open_rasterio(tif_file[0], masked=True).squeeze()  # 2D array
        # sum all valid cells
        total_population = float(da.sum(skipna=True).values)
        # store result
        results.append({"year": year, age_category: int(total_population)})
    
        # 3. Create dataframe
        df = pd.DataFrame(results).sort_values("year").set_index("year")
        df_ages = pd.concat([df_ages, df], axis=1)
    df_years = pd.concat([df_years, df_ages], axis=0)
    
df_years.to_csv('totals_per_age_and_year.csv', sep=',', index=True)
print(df_years)