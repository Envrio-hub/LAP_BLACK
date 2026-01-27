import requests
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp
from decimal import Decimal, ROUND_HALF_UP

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
    "end_timestamp": int(datetime(2050,12,31,23,30).timestamp())
}

data = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)

df = pd.DataFrame.from_dict(data.json())
df = df.rename(columns={"value":"tp"})

df['date_time'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
df = df.drop(columns=('timestamp'))
df['tp'] = df['tp']*86400

stats = StatisticalTools(data_frame=df, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

spi = stats.compute_spi(scale=12)

period_labels = ["1971–2000", "1981–2010", "1991–2020"]
period_I = spi[(spi.index>pd.to_datetime('1970-12-31', utc=True)) & (spi.index<pd.to_datetime('2001-01-01', utc=True))]
period_II = spi[(spi.index>pd.to_datetime('1980-12-31', utc=True)) & (spi.index<pd.to_datetime('2011-1-1', utc=True))]
period_III = spi[(spi.index>pd.to_datetime('1990-12-31', utc=True)) & (spi.index<pd.to_datetime('2021-1-1', utc=True))]
period_IV = spi[spi.index>pd.to_datetime('2019-01-01', utc=True)]

periods = [period_I, period_II, period_III]

# Create CDFs
cdf_list = []
for period in periods:
    sorted_vals = np.sort(period.dropna().values)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    cdf_list.append(pd.DataFrame(data={'values':sorted_vals, 'cdf':cdf}))

cdf_pairs = [(cdf_list[0]['values'], cdf_list[1]['values']),
             (cdf_list[0]['values'], cdf_list[2]['values']),
             (cdf_list[1]['values'], cdf_list[2]['values'])]

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

comparison_periods = ['1971–2000 vs 1981–2010', '1971–2000 vs 1991–2020', '1981–2010 vs 1991–2020']
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
plt.figure(figsize=(10, 6))

for (cdf, label, (p10, p90), color) in zip(cdf_list, period_labels, percentiles, colors):

    plt.plot(cdf['values'], cdf['cdf'], label=label, color=color, linewidth=1.5)
    plt.axvline(p10,  linestyle="--", color=color, alpha=0.6)
    plt.axvline(p90, linestyle="--", color=color, alpha=0.6)

plt.xlabel("SPI", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative probability", fontsize=12, fontweight='bold')
plt.title("SPI-24 CDFs with 5th and 95th percentiles", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

annual_counts = []

five_year_block_counts = []

anchors = [1971, 1981, 1991]

for i, p, anchor in zip(range(0,len(periods)), periods, anchors):

    # ----------------------------------------------------
    # 1. Define the 5th percentile threshold
    # ----------------------------------------------------
    below_5th = p<percentiles[i][0]

    # ----------------------------------------------------
    # 2. Annual counts of SPI events below the 5th percentile
    # ----------------------------------------------------
    # Boolean mask: True when SPI is below the 5th percentile
    # Count per calendar year
    annual = below_5th.groupby(below_5th.index.year).sum().astype(int)
    annual.name = 'count_below_5th'
    annual_counts.append(annual)

    print("\nAnnual counts of SPI period_{i} < 5th percentile:")
    print(annual)

    # ----------------------------------------------------
    # 3. Non-overlapping 5-year block counts of SPI < 5th percentile
    #    Example blocks: 1980–1984, 1985–1989, ...
    # ----------------------------------------------------
    years = below_5th.index.year

    anchor = 1991  # first block starts at 1991–1995

    # Compute block index: 0 for 1991–1995, 1 for 1996–2000, etc.
    block_index = (years - anchor) // 5

    # Define a "block start year" for each observation, e.g.:
    # 1980–1984 -> 1980
    # 1985–1989 -> 1985
    # 1990–1994 -> 1990, etc.
    block_start_year = anchor + 5 * block_index

    # Put into a Series for grouping
    block_counts = below_5th.groupby(block_start_year).sum().astype(int)
    block_counts.name = 'count_below_5th_5yr_block'

    print("\n5-year block counts of SPI < 5th percentile (non-overlapping):")
    print(block_counts)

    # ----------------------------------------------------
    # 4. (Optional) Reformat 5-year block labels as strings "YYYY–YYYY"
    # ----------------------------------------------------
    block_counts_labeled = block_counts.copy()
    block_counts_labeled.index = [
        f"{start_year}-{start_year+4}" for start_year in block_counts.index
    ]
    five_year_block_counts.append(block_counts_labeled)

    print("\n5-year block counts with labeled intervals:")
    print(block_counts_labeled)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), constrained_layout=True)

for ax, pl, block_count in zip(axes, period_labels, five_year_block_counts):
    x_pos = np.arange(len(block_count))

    ax.plot(x_pos, block_count.values, marker='o')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(block_count.index, rotation=45, ha='right')

    ax.set_title(f"{pl}")
    ax.set_ylabel("Event count")
    ax.grid(True, linestyle='--', alpha=0.5)

plt.show()

# Access period IV statistics
years = ['2021', '2022', '2023', '2024']
percentiles_IV = pd.DataFrame()
for year in years:
    period_year = period_IV[(period_IV.index.year==int(year))]
    sorted_vals = np.sort(period_year.dropna().values)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    cdf_list.append(pd.DataFrame(data={'values':sorted_vals, 'cdf':cdf}))

    p10  = np.percentile(sorted_vals, 10)
    p90 = np.percentile(sorted_vals, 90)

    percentiles_IV = pd.concat([percentiles_IV, pd.DataFrame(data={'Year':year, '10th':p10, '90th':p90, 'CDF':0.10}, index=[year])], axis=0)

    print(f'Year {year}: SPI 10th percentile value = {p10 :.3f}')
    print(f'Year {year}: SPI 90th percentile value = {p90 :.3f}')

# 7. Plot 2021-2024 over 1991-2020 baseline period.

plt.step(cdf_list[2]['values'], cdf_list[2]['cdf'], where='post', label='1991–2020 CDF', color='tab:blue')
plt.axvline(percentiles[2][0],  linestyle="--", color='tab:blue', alpha=0.6)

# Recent years on same CDF
plt.scatter(percentiles_IV["10th"],
            percentiles_IV['CDF'],
            label='2021–2024',
            marker='o',
            color='tab:orange')

offsets = [(-12,7), (5,-10), (-15,7), (-30,-5)]
# Annotate each point with its year
for (_, row), offset in zip(percentiles_IV.iterrows(), offsets):
    plt.annotate(
        text=str(row["Year"]),              # text label
        xy=(row["10th"], row["CDF"]),        # point to label
        xytext=offset,                      # offset in pixels
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel("SPI-6", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative Probability", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print()
