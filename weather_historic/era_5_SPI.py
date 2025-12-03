import requests
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statistical_tools import StatisticalTools

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

period_I = spi[(spi.index>datetime(1970,12,31)) & (spi.index<datetime(2001,1,1))]
period_II = spi[(spi.index>datetime(1980,12,31)) & (spi.index<datetime(2011,1,1))]
period_III = spi[(spi.index>datetime(1990,12,31)) & (spi.index<datetime(2021,1,1))]

periods = [period_I, period_II, period_III]

# Compute percentiles
percentiles = []
period_labels = ["1971–2000", "1981–2010", "1991–2020"]
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

print()
