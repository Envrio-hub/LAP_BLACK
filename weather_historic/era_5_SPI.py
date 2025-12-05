import requests
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp

# Calculate Relative Humidity from ERA5 single level

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

longitude = 24.40
latitude = 40.93
start_timestamp = 0
end_timestamp =  1735680000.0

# Gettin Air Temperature data
measurement = "tp"

params = {
    "measurement": measurement,
    "longitude": longitude,
    "latitude": latitude,
    "start_timestamp": start_timestamp,
    "end_timestamp": end_timestamp
}

start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)
print(f'Duration: {datetime.now()-start}')
df = pd.DataFrame(data={"tp":[x*1000 for x in response.json()['data']['value']],
                        'date_time':[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]},
                  index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None

stats = StatisticalTools(data_frame=df, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

spi = stats.compute_spi()

period_labels = ["1971–2000", "1981–2010", "1991–2020"]
period_I = spi[(spi.index>datetime(1970,12,31)) & (spi.index<datetime(2001,1,1))]
period_II = spi[(spi.index>datetime(1980,12,31)) & (spi.index<datetime(2011,1,1))]
period_III = spi[(spi.index>datetime(1990,12,31)) & (spi.index<datetime(2021,1,1))]

periods = [period_I, period_II, period_III]
period_pairs= [(period_I, period_II), (period_I, period_III), (period_II, period_III)]

Kolmogorov_Smirnov = pd.DataFrame()
Anderson_Darling = pd.DataFrame()
for label, pair in zip(period_labels, period_pairs):
    stat, p = ks_2samp(pair[0], pair[1])
    Kolmogorov_Smirnov = pd.concat([Kolmogorov_Smirnov,
                                    pd.DataFrame(data={'KS_Statistic':round(stat,2),'p_value':round(p,2)}, index=[label])], axis=0)
    result = anderson_ksamp([pair[0], pair[1]])
    print(result.pvalue)
    Anderson_Darling = pd.concat([Anderson_Darling, pd.DataFrame(data={'statistic':result.statistic,
                                                                       'critical_values':result.critical_values,
                                                                       'pvalue':result.pvalue})], axis=0)

# Compute percentiles
percentiles = []
colors = ["tab:blue", "tab:orange", "tab:green"]

for w in periods:
    w = w.dropna().values  # clean array

    # SPI values at the 5th and 95th percentiles
    p5  = np.percentile(w, 5)
    p95 = np.percentile(w, 95)

    # Build ECDF
    sorted_vals = np.sort(w)
    ecdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

    # Cumulative probabilities corresponding to p5 and p95
    # (first ECDF value where SPI >= threshold)
    prob5  = ecdf[sorted_vals >= p5][0]
    prob95 = ecdf[sorted_vals >= p95][0]

    percentiles.append((p5, p95, prob5, prob95))

# Print table of results
for label, (p5, p95, prob5, prob95) in zip(period_labels, percentiles):
    print(f"{label}:")
    print(f"  SPI 5th  percentile value = {p5:.3f}, CDF ≈ {prob5:.3f}")
    print(f"  SPI 95th percentile value = {p95:.3f}, CDF ≈ {prob95:.3f}")
    print()

tails_5th = [(period_I[period_I<percentiles[0][0]], period_II[period_II<percentiles[1][0]]),
             (period_I[period_I<percentiles[0][0]], period_III[period_III<percentiles[2][0]]),
             (period_II[period_II<percentiles[1][0]], period_III[period_III<percentiles[2][0]])]

Kolmogorov_Smirnov_5th = pd.DataFrame()
Anderson_Darling_5th = pd.DataFrame()
for label, pair in zip(period_labels, tails_5th):
    stat, p = ks_2samp(pair[0], pair[1])
    Kolmogorov_Smirnov_5th = pd.concat([Kolmogorov_Smirnov_5th,
                                    pd.DataFrame(data={'KS_Statistic':[round(stat,2)],'p_value':[round(p,2)]}, index=[label])], axis=0)
    result = anderson_ksamp([pair[0], pair[1]])
    print(result.statistic)
    Anderson_Darling_5th = pd.concat([Anderson_Darling_5th, pd.DataFrame(data={'AD_Statistic':[result]}, index=[label])], axis=0)

tails_95th = [(period_I[period_I<percentiles[0][1]], period_II[period_II<percentiles[1][1]]),
             (period_I[period_I<percentiles[0][1]], period_III[period_III<percentiles[2][1]]),
             (period_II[period_II<percentiles[1][1]], period_III[period_III<percentiles[2][1]])]

Kolmogorov_Smirnov_95th = pd.DataFrame()
Anderson_Darling_95th = pd.DataFrame()
for label, pair in zip(period_labels, tails_95th):
    stat, p = ks_2samp(pair[0], pair[1])
    Kolmogorov_Smirnov_95th = pd.concat([Kolmogorov_Smirnov_95th,
                                    pd.DataFrame(data={'KS_Statistic':[round(stat,2)],'p_value':[round(p,2)]}, index=[label])], axis=0)
    result = anderson_ksamp([pair[0], pair[1]])
    print(result)
    Anderson_Darling_95th = pd.concat([Anderson_Darling_95th, pd.DataFrame(data={'AD_Statistic':[result]}, index=[label])], axis=0)

# Plot ECDFs with vertical percentile lines
plt.figure(figsize=(10, 6))

for (w_series, label, (p5, p95, prob5, prob95), color) in zip(periods, period_labels, percentiles, colors):
    w = w_series.dropna().values
    sorted_vals = np.sort(w)
    ecdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

    plt.plot(sorted_vals, ecdf, label=label, color=color, linewidth=1.5)
    plt.axvline(p5,  linestyle="--", color=color, alpha=0.6)
    plt.axvline(p95, linestyle="--", color=color, alpha=0.6)

plt.xlabel("SPI")
plt.ylabel("Cumulative probability")
plt.title("SPI ECDFs with 5th and 95th percentiles")
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

print()
