import requests
import numpy as np
from datetime import datetime
import pandas as pd
from statistical_tools import StatisticalTools
from scipy.stats import ks_2samp, anderson_ksamp
import matplotlib.pyplot as plt
from spei import si

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
        values = np.array(response.json()['tp']['data']['value'])*1000
        df_hourly = pd.DataFrame(data={measurement: values}, index=pd.to_datetime(timestamps, unit='s', utc=True))
        df_daily = df_hourly.resample('D').sum()
        df = pd.concat([df, df_daily], axis=1)

df['date_time'] = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M', utc=True)

stats = StatisticalTools(data_frame=df, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

spei = stats.compute_spei(scale=1)
spei.index = pd.to_datetime(spei.index, utc=True)

period_labels = ["1971–2000", "1981–2010", "1991–2020"]
period_I = spei[(spei.index>pd.to_datetime('1970-12-31', utc=True)) & (spei.index<pd.to_datetime('2001-01-01', utc=True))]
period_II = spei[(spei.index>pd.to_datetime('1980-12-31', utc=True)) & (spei.index<pd.to_datetime('2011-1-1', utc=True))]
period_III = spei[(spei.index>pd.to_datetime('1990-12-31', utc=True)) & (spei.index<pd.to_datetime('2021-1-1', utc=True))]
period_IV = spei[spei.index>pd.to_datetime('2019-01-01', utc=True)]

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

plt.xlabel("SPEI", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative probability", fontsize=12, fontweight='bold')
plt.title("SPEI-24 CDFs with 10th and 90th percentiles", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
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

plt.xlabel("SPEI-6", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative Probability", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print()
