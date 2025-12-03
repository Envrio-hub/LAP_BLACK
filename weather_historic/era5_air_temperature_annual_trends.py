import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}
measurement = "t2m"

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

data = pd.DataFrame(data={"values":[round(x-273.15,1) for x in response.json()['data']['value']]},
                    index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None
print(f'Duration: {datetime.now()-start}')

df_annual = pd.DataFrame(data={"mean":data['values'].resample('Y').mean().values,"max":data['values'].resample('Y').max().values,
                               "min":data['values'].resample('Y').min().values,'median':data['values'].resample('Y').median().values},
                         index = data.index.year.unique())

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_annual.index, df_annual['mean'], marker='o')
ax.plot(df_annual.index, df_annual['max'], marker='s')
ax.plot(df_annual.index, df_annual['min'], marker='^')

ax.set_ylabel('Temperature (°C)', fontsize='12', fontweight='bold')
ax.legend()
ax.grid(True)

# Ensure years are numeric
x = np.array(df_annual.index, dtype=float)
y = df_annual['mean'].values

# Linear regression: y = m*x + b
m, b = np.polyfit(x, y, 1)

print("\nSlope:", round(m,2))
print("\nIntercept:", round(b,2))

# ax.plot(df_annual.index, m * x + b, color='red', linewidth=2, linestyle='--')
# ax.legend()
plt.show()

print()