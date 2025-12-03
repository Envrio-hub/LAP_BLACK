import requests
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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
df = pd.DataFrame(data={"tp":[x*1000 for x in response.json()['data']['value']]},
                  index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None

# Step 1: Compute anomalies
df_monthly = df['tp'].resample('ME').sum().to_frame()

period_I = df_monthly[(df_monthly.index>datetime(1970,12,31)) & (df_monthly.index<datetime(2001,1,1))]
period_II = df_monthly[(df_monthly.index>datetime(1979,12,31)) & (df_monthly.index<datetime(2011,1,1))]
period_III = df_monthly[(df_monthly.index>datetime(1989,12,31)) & (df_monthly.index<datetime(2021,1,1))]

periods = [period_I, period_II, period_III]

wasp_results = []

for period in periods:
    period['clim_mean'] = period['tp'].groupby(period.index.month).transform('mean')
    period['clim_std']  = period['tp'].groupby(period.index.month).transform('std')

    period['A'] = (period['tp'] - period['clim_mean']) / period['clim_std']

    # Step 2: Define weights (example for 6-month WASP)
    weights = np.array([6,5,4,3,2,1], dtype=float)
    weights /= weights.sum()

    # Step 3: Compute the weighted rolling anomaly

    period['WASP'] = (
        period['A']
        .rolling(6)
        .apply(lambda x: np.sum(weights[::-1] * x), raw=True)
    )

    wasp_results.append(period)

# Unpack WASP time series
wasp_I   = wasp_results[0]["WASP"].dropna()
wasp_II  = wasp_results[1]["WASP"].dropna()
wasp_III = wasp_results[2]["WASP"].dropna()

period_labels = ["1971–2000", "1981–2010", "1991–2020"]
wasp_series = [wasp_I, wasp_II, wasp_III]

# Compute percentiles
percentiles = []
for w in wasp_series:
    p5  = np.percentile(w, 5)
    p95 = np.percentile(w, 95)
    percentiles.append((p5, p95))

# Print results
for label, (p5, p95) in zip(period_labels, percentiles):
    print(f"{label}: 5th percentile = {p5:.3f}, 95th percentile = {p95:.3f}")
plt.figure(figsize=(10, 6))

colors = ["tab:blue", "tab:orange", "tab:green"]

for (w, label, (p5, p95), color) in zip(wasp_series, period_labels, percentiles, colors):
    sorted_vals = np.sort(w)
    ecdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

    plt.plot(sorted_vals, ecdf, label=label, color=color, linewidth=1.5)

    # Add percentile annotations
    plt.axvline(p5,  linestyle="--", color=color, alpha=0.6)
    plt.axvline(p95, linestyle="--", color=color, alpha=0.6)

plt.xlabel("WASP value")
plt.ylabel("Cumulative Probability")
plt.title("ECDF with 5th and 95th Percentiles for WASP")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print()
