import numpy as np
from datetime import datetime
import pandas as pd
from scipy.stats import genextreme
import requests
import matplotlib.pyplot as plt

# Obtain Historic data
base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

longitude = 24.40
latitude = 40.93
start_timestamps = [int(datetime(1991,1,1).timestamp())]
end_timestamps = [int(datetime(2020,12,31).timestamp())]

# Gettin Air Temperature data
measurement = ["tp"]

params = {
    "measurements": measurement,
    "longitude": longitude,
    "latitude": latitude,
    "start_timestamp": start_timestamps[0],
    "end_timestamp": end_timestamps[0]
}

start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)
print(f'Duration: {datetime.now()-start}')
era5_data = pd.DataFrame(data={"tp": np.array(response.json()['tp']['data']['value'])*1000},
                               index=pd.to_datetime(response.json()['tp']['data']['timestamp'], unit='s', utc=True)) if response.status_code==200 else None

reference = era5_data.resample('YE').max().values

# Obtain forcast data

projection_id = 2

location_id = 4445

data_product_ids = [4] # 1 maxTemp 2minTemp 3hurs 4Precipitation 5rsds 6sfcWind

base_url = 'https://envrio.org/cordex_api'

# Step 1 - User authendication
auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr", "password":"TestPass123@123"})

if auth.json().get('access_token'):
    token = auth.json()['access_token']
    headers = {
        "Authorization": f"Bearer {token}"
    }

params = {
    "data_product_ids": data_product_ids,
    "projection_id": projection_id,
    "location_id": location_id,
    "start_timestamp": int(datetime(2021,1,1).timestamp()),
    "end_timestamp": int(datetime(2100,12,31,23,30).timestamp())
}

cordex_data = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)

cordex_df = pd.DataFrame.from_dict(cordex_data.json())
cordex_df = cordex_df.rename(columns={"value":"tp"})

cordex_df.index = pd.to_datetime(cordex_df['timestamp'], unit='s', utc=True)
cordex_df = cordex_df.drop(columns=('timestamp'))
cordex_df['tp'] = cordex_df['tp']*86400

near_future = cordex_df[cordex_df.index<=pd.to_datetime('2050-12-31', utc=True)].resample('YE').max()
far_future = cordex_df[cordex_df.index>=pd.to_datetime('2071-1-1', utc=True)].resample('YE').max().values

periods = [reference, near_future, far_future]
labels  = ["1991–2020", "2021–2050", "2071–2100"]

gev_parameters_list = []

# ---- Fit GEV for each period ----
for period in periods:
    period = np.asarray(period['tp'], dtype=float)
    period = period[np.isfinite(period)]  # safety

    c, loc, scale = genextreme.fit(period)  # SciPy: c = -xi
    xi = -c
    mu = loc
    sigma = scale

    gev_parameters_list.append((mu, sigma, xi))  # <-- tuple


def gev_cdf(x, mu, sigma, xi):
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if np.isclose(xi, 0.0):
        z = (x - mu) / sigma
        return np.exp(-np.exp(-z))

    t = 1.0 + xi * (x - mu) / sigma

    # Initialize
    F = np.empty_like(x, dtype=float)

    # Outside support
    if xi > 0:
        F[t <= 0] = 0.0
    else:  # xi < 0
        F[t <= 0] = 1.0

    # Inside support
    mask = t > 0
    F[mask] = np.exp(-(t[mask]) ** (-1.0 / xi))
    return F


# ---- Common grid for all periods (recommended for comparability) ----
global_max = max(np.max(p['tp']) for p in periods)
xgrid = np.linspace(0, global_max * 1.2, 600)

# ---- Plot all CDFs together ----
plt.figure(figsize=(10, 6))

for (mu, sigma, xi), label in zip(gev_parameters_list, labels):
    Fgrid = gev_cdf(xgrid, mu, sigma, xi)
    plt.plot(xgrid, Fgrid, linewidth=2, label=label)

plt.xlabel("Annual maximum daily precipitation", fontsize=12, fontweight="bold")
plt.ylabel("Cumulative probability", fontsize=12, fontweight="bold")
plt.title("GEV CDFs of Annual Block Maxima (All Periods)", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print()