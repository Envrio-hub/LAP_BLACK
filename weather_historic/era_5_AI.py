import requests
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp

# Helper function to calculate aridity index

def calculate_cdf(data):
    """Compute the empirical cumulative distribution function (CDF) for a 1D array of data."""
    sorted_data = np.sort(data)
    aranged_data = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, aranged_data

def make_ecdf(data):
    """
    Create an empirical CDF from a 1D array-like of samples.
    Returns:
      cdf_func: function that evaluates the ECDF at given x
      x_sorted: sorted data (for plotting if needed)
    """
    # Convert to NumPy array and drop NaNs if needed
    x = np.asarray(data)
    x = x[~np.isnan(x)]
    
    # Sort data
    x_sorted = np.sort(x)
    n = x_sorted.size

    def cdf_func(v):
        """
        ECDF evaluated at v (scalar or array).
        Returns P(X <= v) under the empirical distribution.
        """
        v = np.asarray(v)
        # searchsorted gives index where v would be inserted to keep order
        idx = np.searchsorted(x_sorted, v, side="right")
        return idx / n

    return cdf_func, x_sorted

# Calculate Relative Humidity from ERA5 single level

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

longitude = 24.40
latitude = 40.93
start_timestamps = [0, datetime(1990,1,1,0).timestamp(), datetime(2010,1,1,0).timestamp()]
end_timestamps =  [datetime(1989,12,31,23,30).timestamp(), datetime(2009,12,31,23,30).timestamp(), datetime(2024,12,31,23,30).timestamp()]
elev = 45

df = pd.DataFrame()

for start_timestamp, end_timestamp in zip(start_timestamps, end_timestamps):

    params = {
        "longitude": longitude,
        "latitude": latitude,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "elev":elev
    }

    start = datetime.now()
    response = requests.get(f'{base_url}/evapotranspiration', headers=headers, params=params)
    print(f'ETo series created in : {datetime.now()-start}')

    df = pd.concat([df, pd.DataFrame({"pet":response.json()['ET_Daily']['values']},
                                     index=pd.to_datetime(response.json()['ET_Daily']['timestamp'], unit='s', utc=True))],
                   axis=0)

# Gettin Air Temperature data
measurements = ["tp"]


for measurement in measurements:
    params = {
        "measurements": measurement,
        "longitude": longitude,
        "latitude": latitude,
        "start_timestamp": start_timestamps[0],
        "end_timestamp": end_timestamps[2]
    }

    start = datetime.now()
    response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)
    print(f'Duration: {datetime.now()-start}')
    if response.status_code == 200:
        timestamps = response.json()['tp']['data']['timestamp']
        values = response.json()['tp']['data']['value']
        df_hourly = pd.DataFrame(data={measurement: values}, index=pd.to_datetime(timestamps, unit='s', utc=True))
        df_daily = df_hourly.resample('D').sum()
        df = pd.concat([df, df_daily], axis=1)

df['date_time'] = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M', utc=True)

stats = StatisticalTools(data_frame=df, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

ai = stats.aridity_index(pet_col='pet')

period_labels = ["1971–2000 vs 1981–2010", "1981–2010 vs 1991–2020", "1971–2000 vs 1991–2020"]
period_I = ai[(ai.index>pd.to_datetime('1970-12-31', utc=True)) & (ai.index<pd.to_datetime('2001-01-01', utc=True))]
period_II = ai[(ai.index>pd.to_datetime('1980-12-31', utc=True)) & (ai.index<pd.to_datetime('2011-1-1', utc=True))]
period_III = ai[(ai.index>pd.to_datetime('1990-12-31', utc=True)) & (ai.index<pd.to_datetime('2021-1-1', utc=True))]

cdf_I_x, cdf_I_y = calculate_cdf(period_I.dropna().values)
cdf_II_x, cdf_II_y = calculate_cdf(period_II.dropna().values)
cdf_III_x, cdf_III_y = calculate_cdf(period_III.dropna().values)

cdfs = [(cdf_I_x, cdf_I_y), (cdf_II_x, cdf_II_y), (cdf_III_x, cdf_III_y)]
cdf_pairs= [(cdf_I_x, cdf_II_x), (cdf_I_x, cdf_III_x), (cdf_II_x, cdf_III_x)]

# Compute percentiles
percentiles = []
colors = ["tab:blue", "tab:orange", "tab:green"]

for w in cdfs:

    # AI values at the 5th and 95th percentiles
    p10  = np.percentile(w[0], 10)
    p90 = np.percentile(w[0], 90)

    # Cumulative probabilities corresponding to p5 and p95
    # (first ECDF value where AI >= threshold)
    prob10  = w[1][w[0] >= p10][0]
    prob90 = w[1][w[0] >= p90][0]

    percentiles.append((p10, p90, prob10, prob90))

# Print table of results
for label, (p10, p90, prob10, prob90) in zip(period_labels, percentiles):
    print(f"\n{label}:")
    print(f"  AI 10th  percentile value = {p10:.3f}, CDF ≈ {prob10:.3f}")
    print(f"  AI 90th percentile value = {p90:.3f}, CDF ≈ {prob90:.3f}\n")

Kolmogorov_Smirnov = pd.DataFrame()
Anderson_Darling = pd.DataFrame()
for label, pair in zip(period_labels, cdf_pairs):
    stat, p = ks_2samp(pair[0], pair[1])
    Kolmogorov_Smirnov = pd.concat([Kolmogorov_Smirnov,
                                    pd.DataFrame(data={'KS_Statistic':round(stat,2),'p_value':round(p,4)}, index=[label])], axis=0)
    result = anderson_ksamp([pair[0], pair[1]])
    print(result.pvalue)
    Anderson_Darling = pd.concat([Anderson_Darling, pd.DataFrame(data={'statistic':result.statistic,
                                                                       'critical_values':result.critical_values,
                                                                       'pvalue':result.pvalue})], axis=0)


# Plot ECDFs with vertical percentile lines
plt.figure(figsize=(10, 6))

thirty_years_period_labels = ["1971–2000", "1981–2010", "1991–2020"]

for (w_series, label, (p10, p90, prob10, prob90), color) in zip(cdfs, thirty_years_period_labels, percentiles, colors):
    plt.plot(w_series[0], w_series[1], label=label, color=color, linewidth=1.5)
    plt.axvline(p10,  linestyle="--", color=color, alpha=0.6)
    plt.axvline(p90, linestyle="--", color=color, alpha=0.6)

plt.xlabel("Aridity Index", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative probability", fontsize=12, fontweight='bold')
plt.title("AI CDFs with 10th and 90th percentiles", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print()

# Project 2021-2024 period over 1991-2020
period_IV = ai[ai.index>pd.to_datetime('2020-12-31', utc=True)]

ecdf_base, x_base_sorted = make_ecdf(period_III.values)

probs_recent = ecdf_base(period_IV.values)

for year, ai_val, p in zip(period_IV.index, period_IV.values, probs_recent):
    print(year, ai_val, p)

# Baseline CDF points
n_base = x_base_sorted.size
y_base = np.arange(1, n_base + 1) / n_base

plt.step(x_base_sorted, y_base, where='post', label='1991–2020 CDF', color='tab:blue')

# Recent years on same CDF
plt.scatter(period_IV.values,
            ecdf_base(period_IV.values),
            label='2021–2024',
            marker='o',
            color='tab:orange')

plt.xlabel("Aridity Index", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative Probability", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()