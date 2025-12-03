import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}
measurement = "tp"

longitude = 24.40
latitude = 40.93
start_timestamp = 0
end_timestamp =  1735680000.0

params = {
    "measurement": measurement,
    "longitude": longitude,
    "latitude": latitude,
    "start_timestamp": start_timestamp,
    "end_timestamp": end_timestamp
}

start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)

data = pd.DataFrame(data={"values":[x*1000 for x in response.json()['data']['value']]},
                    index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None
print(f'Duration: {datetime.now()-start}')

df_daily = data['values'].resample('D').sum().to_frame()
df_daily['heavy_precipitation'] = df_daily['values'] >= 50
df_daily['year'] = df_daily.index.year

grouped = df_daily.groupby('year')

heavy_precipitation = [group[1]['heavy_precipitation'].sum() for group in grouped]

df_annual = pd.DataFrame(data={"sum":[round(x,0) for x in data['values'].resample('Y').sum().values],
                               "max":[round(x,1) for x in data['values'].resample('Y').max().values],
                               "heavy_precipitation":heavy_precipitation},
                         index = data.index.year.unique())

# Create a figure with 2 rows, 1 column of axes, sharing the x-axis (years)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# Common x as numeric years
x = np.array(df_annual.index, dtype=float)

# ==============
# 1) SUM GRAPH
# ==============

y_sum = df_annual['sum'].values

ax1.plot(df_annual.index, df_annual['sum'], marker='o', label='Annual total')

ax1.set_ylabel('Precipitation (mm)', fontsize=10, fontweight='bold')
ax1.grid(True)

# Linear regression for sum
m_sum, b_sum = np.polyfit(x, y_sum, 1)

print("\nSUM series:")
print("Slope:", round(m_sum, 2))
print("Intercept:", round(b_sum, 2))

ax1.plot(df_annual.index,
         m_sum * x + b_sum,
         color='red',
         linewidth=2,
         linestyle='--',
         label='Trend (sum)')

ax1.text(0.75, 0.90,
         f'Slope (a) = {round(m_sum, 2)}',
         transform=ax1.transAxes,
         fontsize=12)

# ==============
# 2) MAX GRAPH
# ==============

y_max = df_annual['max'].values

ax2.plot(df_annual.index, df_annual['max'], marker='o', label='Annual max')

ax2.set_ylabel('Precipitation Intensity (mm/hr)', fontsize=10, fontweight='bold')
ax2.grid(True)

# Linear regression for max
m_max, b_max = np.polyfit(x, y_max, 1)

print("\nMAX series:")
print("Slope:", round(m_max, 2))
print("Intercept:", round(b_max, 2))

ax2.plot(df_annual.index,
         m_max * x + b_max,
         color='red',
         linewidth=2,
         linestyle='--',
         label='Trend (max)')

ax2.text(0.75, 0.90,
         f'Slope (a) = {round(m_max, 2)}',
         transform=ax2.transAxes,
         fontsize=12)

# ==============
# 3) EXTREME GRAPH
# ==============

y_extreme = df_annual['heavy_precipitation'].values

ax3.plot(df_annual.index, df_annual['heavy_precipitation'], marker='o', label='Annual max')

ax3.set_ylabel('Heavy Precipitation (>50 mm/d)', fontsize=10, fontweight='bold')
ax3.grid(True)
ax3.set_yticks(range(0, 4))

# Linear regression for max
m_max, b_max = np.polyfit(x, y_extreme, 1)

print("\nMAX series:")
print("Slope:", round(m_max, 2))
print("Intercept:", round(b_max, 2))

ax3.plot(df_annual.index,
         m_max * x + b_max,
         color='red',
         linewidth=2,
         linestyle='--',
         label='Trend (max)')

ax3.text(0.75, 0.90,
         f'Slope (a) = {round(m_max, 2)}',
         transform=ax3.transAxes,
         fontsize=12)

plt.tight_layout()
plt.show()

print()