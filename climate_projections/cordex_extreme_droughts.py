import requests
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp
from decimal import Decimal, ROUND_HALF_UP

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

longitude = 24.40
latitude = 40.93
start_timestamps = [0, datetime(1990,1,1,0).timestamp(), datetime(2010,1,1,0).timestamp()]
end_timestamps =  [datetime(1989,12,31,23,30).timestamp(), datetime(2009,12,31,23,30).timestamp(), datetime(2024,12,31,23,30).timestamp()]


# Gettin Air Temperature data
measurements = ["tp"]

params = {
    "measurements": measurements,
    "longitude": longitude,
    "latitude": latitude,
    "start_timestamp": start_timestamps[0],
    "end_timestamp": end_timestamps[2]
}

start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)
print(f'Duration: {datetime.now()-start}')
data = pd.DataFrame(data={"tp":np.array(response.json()['tp']['data']['value'])*1000},
                    index=pd.to_datetime(response.json()['tp']['data']['timestamp'], unit='s', utc=True)) if response.status_code==200 else None

# 2. Aggregate hourly to daily precipitation (mm/day)
df_daily = data.resample('D').sum()
df_daily['date'] = df_daily.index

base_url = 'https://envrio.org/cordex_api'

# Step 1 - User authendication
auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr", "password":"TestPass123@123"})

if auth.json().get('access_token'):
    token = auth.json()['access_token']
    headers = {
        "Authorization": f"Bearer {token}"
    }

# Step 2 - Get location id
long = Decimal(24.394194).quantize(Decimal('0.000000'), ROUND_HALF_UP)
lat = Decimal(40.936811).quantize(Decimal('0.000000'), ROUND_HALF_UP)

params = {
    "longitude": long,
    "latitude": lat
    }

nearest_data_point = requests.get(f'{base_url}/nearest_data_point', headers=headers, params=params)

if nearest_data_point.json().get('Location ID'):
    location_id = nearest_data_point.json()['Location ID']
    print(f'\nLocation ID: {location_id}\n')
    location_id = 3726

# Step 3 - Select a projection
projections = requests.get(f'{base_url}/projections', headers=headers)
for projection in projections.json():
    print(projection)

projection_id = 1

# Step 4 - Select Data Products
data_products = requests.get(f'{base_url}/data_products', headers=headers)
for product in data_products.json():
    print(product)

data_product_ids = [4] # Precipitation

# Step 5 - Get data series

params = {
    "data_product_ids": data_product_ids,
    "projection_id": projection_id,
    "location_id": location_id,
    "start_timestamp": int(datetime(2021,1,1).timestamp()),
    "end_timestamp": int(datetime(2100,12,31,23,30).timestamp())
}

cordex_data = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)

cordex_df = pd.DataFrame.from_dict(cordex_data.json())
cordex_df = cordex_df.rename(columns={"value":"tp"})

cordex_df['date'] = pd.to_datetime(cordex_df['timestamp'], unit='s', utc=True)
cordex_df = cordex_df.drop(columns=('timestamp'))
cordex_df.index = cordex_df['date']
cordex_df['tp'] = cordex_df['tp']*86400

# 2. Divide into climate normals periods 1971-2000, 1981-2010, 1991-2020.

period_labels = ["1991-2020", "2021-2050", "2071-2100"]
reference = df_daily[(df_daily.index>pd.to_datetime('1990-12-31', utc=True)) & (df_daily.index<pd.to_datetime('2021-1-1', utc=True))]
near_future = cordex_df[cordex_df.index<=pd.to_datetime('2050-12-31', utc=True)]
far_future = cordex_df[cordex_df.index>=pd.to_datetime('2071-1-1', utc=True)]
periods = [reference, near_future, far_future]

# 3. Calculate extreme droughts based on maximum consecutive dry days (CDD)
inv_cdf_list = []
for period, label in zip(periods, period_labels):
    stats = StatisticalTools(data_frame=period, date_format='%Y-%m-%d', precip_col='tp')
    cdd_lengths, cdf, inv_csf = stats.extreme_drought(dry_threshold=1.0)
    inv_cdf_list.append(inv_csf)

# 4. Compute percentiles
percentiles = []
colors = ["tab:blue", "tab:orange", "tab:green"]

for w in inv_cdf_list:

    # Cumulative probabilities corresponding to p5
    # (first CDF value where EP < threshold)
    p5  = np.min(w[w['ExceedanceProb'] < 5/100]['CDD_length'].iloc[0])

    percentiles.append((p5, 0.05))

# Print table of results
for label, (p5,prob5) in zip(period_labels, percentiles):
    print(f"{label}:")
    print(f" ED 5th  percentile value = {p5:.3f}, CDF ≈ {prob5:.3f}\n")
    print()

# 5. Assess statistical differences between periods
inv_csf_pairs= [(inv_cdf_list[0], inv_cdf_list[1]), (inv_cdf_list[0], inv_cdf_list[2]), (inv_cdf_list[1], inv_cdf_list[2])]
comparing_periods = ["1991–2020 vs 2021–2050", "1991–2020 vs 2071–2100"]
Kolmogorov_Smirnov = pd.DataFrame()
Anderson_Darling = pd.DataFrame()
for label, pair in zip(comparing_periods, inv_csf_pairs):
    stat, p = ks_2samp(pair[0]['CDD_length'], pair[1]['CDD_length'])
    Kolmogorov_Smirnov = pd.concat([Kolmogorov_Smirnov,
                                    pd.DataFrame(data={'KS_Statistic':round(stat,2),'p_value':round(p,2)}, index=[label])], axis=0)
    result = anderson_ksamp([pair[0]['CDD_length'], pair[1]['CDD_length']])
    print(result.pvalue)
    Anderson_Darling = pd.concat([Anderson_Darling, pd.DataFrame(data={'statistic':result.statistic,
                                                                       'critical_values':result.critical_values,
                                                                       'pvalue':result.pvalue})], axis=0)

# 6. Plot inverse CDFs for each period

plt.figure(figsize=(10, 6))
thirty_years_period_labels = ["1971–2000", "1981–2010", "1991–2020"]
for (w_series, label, (p5, prob5), color) in zip(inv_cdf_list, thirty_years_period_labels, percentiles, colors):
    plt.plot(w_series['CDD_length'], w_series['ExceedanceProb'], label=label, color=color, linewidth=1.5)
    plt.axvline(p5,  linestyle="--", color=color, alpha=0.6)

plt.xlabel("CDD length (days)", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative probability", fontsize=12, fontweight='bold')
plt.title("Inverse CDF (Exceedance Probability)", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7. Calculate extreme droughts for 2021-2024.
stats_IV = StatisticalTools(data_frame=period_IV, date_format='%Y-%m-%d', precip_col='tp')
years = ['2021', '2022', '2023', '2024']
percentiles_IV = pd.DataFrame()
for year in years:
    period_year = period_IV[(period_IV.index.year==int(year))]
    stats_year = StatisticalTools(data_frame=period_year, date_format='%Y-%m-%d', precip_col='tp')
    cdd_lengths, cdf, inv_cdf = stats_year.extreme_drought(dry_threshold=1.0)

    p5  = inv_cdf[inv_cdf['ExceedanceProb'] < 5/100]['CDD_length'].iloc[0]

    percentiles_IV = pd.concat([percentiles_IV, pd.DataFrame(data={'Year':year, 'CDD':p5, 'CDF':0.05}, index=[year])], axis=0)

    print(f'Year {year}: ED 5th percentile value = {p5:.3f}')

# 7. Plot 2021-2024 over 1991-2020 baseline period.

plt.step(inv_cdf_list[2]['CDD_length'], inv_cdf_list[2]['ExceedanceProb'], where='post', label='1991–2020 CDF', color='tab:blue')
plt.axvline(percentiles[2][0],  linestyle="--", color='tab:blue', alpha=0.6)

# Recent years on same CDF
plt.scatter(percentiles_IV["CDD"],
            percentiles_IV['CDF'],
            label='2021–2024',
            marker='o',
            color='tab:orange')

# Annotate each point with its year
for _, row in percentiles_IV.iterrows():
    plt.annotate(
        text=str(row["Year"]),              # text label
        xy=(row["CDD"], row["CDF"]),        # point to label
        xytext=(5, 5),                      # offset in pixels
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel("CDD length (days)", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative Probability", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print()
