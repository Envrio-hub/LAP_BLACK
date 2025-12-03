import requests
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

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
data = pd.DataFrame(data={"tp":[x*1000 for x in response.json()['data']['value']]},
                    index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None

# 2. Aggregate hourly to daily precipitation (mm/day)
df_daily = data['tp'].resample('D').sum()

# 3. Flag dry days (precip < 1 mm)
dry = (df_daily < 1).astype(int)

# 3. Identify dry-day sequences
# Switch from dry (1) to wet (0) splits sequences. cumsum creates sequence groups.
groups = (dry == 0).cumsum()

# Running length of each dry sequence
cdd = dry.groupby(groups).cumsum()

# 4. Get maximum consecutive dry days per year
max_cdd_yearly = cdd.resample("YE").max()
max_cdd_yearly.index = max_cdd_yearly.index.year  # cleaner x-axis labels

# 5. Linear Regression for sum counts
m_sum, b_sum = np.polyfit(max_cdd_yearly.index, max_cdd_yearly.values, 1)
print('Slope', round(m_sum,1))

# 5. Plot
plt.figure(figsize=(10,5))
plt.plot(max_cdd_yearly.index, max_cdd_yearly.values, marker="o")
plt.title("Maximum Consecutive Dry Days per Year (precip < 1 mm)")
plt.xlabel("Year")
plt.ylabel("Max CDD (days)")
plt.grid(True)
plt.tight_layout()

plt.plot(max_cdd_yearly.index,
         m_sum * max_cdd_yearly.index + b_sum,
         color='red',
         linewidth=2,
         linestyle='--',
         label='Trend (sum)')

plt.show()

print()
