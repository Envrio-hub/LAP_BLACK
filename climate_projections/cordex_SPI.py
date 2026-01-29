import requests
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp
from decimal import Decimal, ROUND_HALF_UP

# Define the SPI time sclale. Could be 1, 6, 24 for meteorological, agriculture and hydrological assessments.

time_scale = 24

# Calculate SPI for base 30-year period 1991-2020 based on ERA5 single levels dataset

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

longitude = 24.40
latitude = 40.93
start_timestamps = [int(datetime(1991,1,1).timestamp())]
end_timestamps = [int(datetime(2020,12,31).timestamp())]

# Gettin Air Temperature data
measurement = ["tp"]

params = {
    "measurements": measurement,
    "longitude": longitude,
    "latitude": latitude,
    "start_timestamp": start_timestamps[0],
    "end_timestamp": end_timestamps[0]
}

start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)
print(f'Duration: {datetime.now()-start}')
era5_data = pd.DataFrame(data={"tp": np.array(response.json()['tp']['data']['value'])*1000,
                               'date_time': pd.to_datetime(response.json()['tp']['data']['timestamp'], unit='s', utc=True)},
                               index=pd.to_datetime(response.json()['tp']['data']['timestamp'], unit='s', utc=True)) if response.status_code==200 else None

stats = StatisticalTools(data_frame=era5_data, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

reference_spi = stats.compute_spi(scale=time_scale)

# Calculate SPI for near and far future based on CORDEX climate projections

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
    location_id = 4445

# Step 3 - Select a projection
projections = requests.get(f'{base_url}/projections', headers=headers)
for projection in projections.json():
    print(projection)

projection_id = 3

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

cordex_df['date_time'] = pd.to_datetime(cordex_df['timestamp'], unit='s', utc=True)
cordex_df = cordex_df.drop(columns=('timestamp'))
cordex_df['tp'] = cordex_df['tp']*86400

stats = StatisticalTools(data_frame=cordex_df, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

cordex_spi = stats.compute_spi(scale=time_scale)

near_future = cordex_spi[cordex_spi.index<=pd.to_datetime('2050-12-31', utc=True)]
far_future = cordex_spi[cordex_spi.index>=pd.to_datetime('2071-1-1', utc=True)]

# Compare the SPI destributions between the reference, near and far future projections.

period_labels = ["1991–2020", "2021-2050", "2071-2100"]

periods = [reference_spi, near_future, far_future]

# Create CDFs
cdf_list = []
for period in periods:
    sorted_vals = np.sort(period.dropna().values)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    cdf_list.append(pd.DataFrame(data={'values':sorted_vals, 'cdf':cdf}))

cdf_pairs = [(cdf_list[0]['values'], cdf_list[1]['values']),
             (cdf_list[0]['values'], cdf_list[2]['values'])]

# Compute percentiles
percentiles = []
colors = ["tab:blue", "tab:orange", "tab:green"]

for cdf in cdf_list:

    # SPI values at the 5th and 95th percentiles
    p10  = np.percentile(cdf['values'], 10)
    p90 = np.percentile(cdf['values'], 90)

    percentiles.append((p10, p90))

# Print table of results
for label, (p10, p90) in zip(period_labels, percentiles):
    print(f"{label}:")
    print(f"  SPI 10th  percentile value = {p10:.3f}, CDF ≈ 0.10")
    print(f"  SPI 90th percentile value = {p90:.3f}, CDF ≈ 0.90\n")
    print()

comparison_periods = ['1991–2020 vs 2021–2050', '1991–2020 vs 2071–2100']
Kolmogorov_Smirnov = pd.DataFrame()
Anderson_Darling = pd.DataFrame()
for label, pair in zip(comparison_periods, cdf_pairs):
    stat, p = ks_2samp(pair[0], pair[1])
    Kolmogorov_Smirnov = pd.concat([Kolmogorov_Smirnov,
                                    pd.DataFrame(data={'KS_Statistic':round(stat,2),'p_value':round(p,2)}, index=[label])], axis=0)
    result = anderson_ksamp([pair[0], pair[1]])
    print(result.pvalue)
    Anderson_Darling = pd.concat([Anderson_Darling, pd.DataFrame(data={'statistic':result.statistic,
                                                                       'critical_values':result.critical_values,
                                                                       'pvalue':result.pvalue})], axis=0)

# Plot ECDFs with vertical percentile lines
cdf10 = pd.read_csv('t24_projectionID3_locationID4445_reference.csv')
cdf11 = pd.read_csv('t24_projectionID3_locationID4445_near.csv')
cdf12 = pd.read_csv('t24_projectionID3_locationID4445_far.csv')
cdf_list_2 = [cdf10, cdf11, cdf12]

cdf00 = pd.read_csv('t24_projectionID1_locationID3726_reference.csv')
cdf01 = pd.read_csv('t24_projectionID1_locationID3726_near.csv')
cdf02 = pd.read_csv('t24_projectionID1_locationID3726_far.csv')
cdf_list_1 = [cdf00, cdf01, cdf02]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

# ---- FIRST SUBPLOT ----
ax = axes[0]

for (cdf, label, (p10, p90), color) in zip(cdf_list_1, period_labels, percentiles, colors):
    ax.plot(cdf['values'], cdf['cdf'], label=label, color=color, linewidth=1.5)
    ax.axvline(p10, linestyle="--", color=color, alpha=0.6)
    ax.axvline(p90, linestyle="--", color=color, alpha=0.6)

ax.set_ylabel("Cumulative probability", fontsize=12, fontweight='bold')
ax.set_title("SPI-24 CDFs RCP8.5-SMHI-RCA4", fontsize=12)
ax.grid(True)
ax.legend()
axes[0].text(
    0.02, 0.95, "(a)",
    transform=axes[0].transAxes,
    fontsize=12,
    fontweight="bold",
    va="top"
)

# ---- SECOND SUBPLOT ----
ax = axes[1]

for (cdf, label, (p10, p90), color) in zip(cdf_list_2, period_labels, percentiles, colors):
    ax.plot(cdf['values'], cdf['cdf'], label=label, color=color, linewidth=1.5)
    ax.axvline(p10, linestyle="--", color=color, alpha=0.6)
    ax.axvline(p90, linestyle="--", color=color, alpha=0.6)

ax.set_xlabel("SPI", fontsize=12, fontweight='bold')
ax.set_ylabel("Cumulative probability", fontsize=12, fontweight='bold')
ax.set_title("SPI-24 CDFs RCP8.5-KNMI-RACMO22E", fontsize=12)
ax.grid(True)
ax.legend()
axes[1].text(
    0.02, 0.95, "(b)",
    transform=axes[1].transAxes,
    fontsize=12,
    fontweight="bold",
    va="top"
)

plt.tight_layout()
plt.show()
print()
