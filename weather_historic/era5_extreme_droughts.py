import requests
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp

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

# 2. Divide into climate normals periods 1971-2000, 1981-2010, 1991-2020.

period_labels = ["1971-2000", "1981-2010", "1991-2020"]
period_I = df_daily[(df_daily.index>pd.to_datetime('1970-12-31', utc=True)) & (df_daily.index<pd.to_datetime('2001-01-01', utc=True))]
period_II = df_daily[(df_daily.index>pd.to_datetime('1980-12-31', utc=True)) & (df_daily.index<pd.to_datetime('2011-1-1', utc=True))]
period_III = df_daily[(df_daily.index>pd.to_datetime('1990-12-31', utc=True)) & (df_daily.index<pd.to_datetime('2021-1-1', utc=True))]
periods = [period_I, period_II, period_III
           ]
# 3. Calculate extreme droughts based on maximum consecutive dry days (CDD)
inv_csf_list = []
for period, label in zip(periods, period_labels):
    stats = StatisticalTools(data_frame=period, date_format='%Y-%m-%d %H:%M', precip_col='tp')
    cdd_lengths, cdf, inv_csf = stats.extreme_drought(dry_threshold=1.0)
    inv_csf_list.append((label, inv_csf))


# 4. Plot inverse CDFs for each period

# plt.figure(figsize=(10, 6))

# thirty_years_period_labels = ["1971–2000", "1981–2010", "1991–2020"]

# for (w_series, label, (p10, p90, prob10, prob90), color) in zip(cdfs, thirty_years_period_labels, percentiles, colors):
#     plt.plot(w_series[0], w_series[1], label=label, color=color, linewidth=1.5)
#     plt.axvline(p10,  linestyle="--", color=color, alpha=0.6)
#     plt.axvline(p90, linestyle="--", color=color, alpha=0.6)

# plt.xlabel("CDD length (days)", fontsize=12, fontweight='bold')
# plt.ylabel("P(CDD > x)", fontsize=12, fontweight='bold')
# plt.title("Inverse CDF (Exceedance Probability)", fontsize=12)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()



print()
