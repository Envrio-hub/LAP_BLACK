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
end_timestamp =  datetime(2024,12,31,23).timestamp() 1735680000.0

# Gettin Air Temperature data
measurement = ["tp"]

params = {
    "measurements": measurement,
    "longitude": longitude,
    "latitude": latitude,
    "start_timestamp": start_timestamp,
    "end_timestamp": end_timestamp
}

start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)
print(f'Duration: {datetime.now()-start}')
df = pd.DataFrame(data={"tp":[x*1000 for x in response.json()[measurement[0]]['data']['value']],
                        'date_time':pd.to_datetime(response.json()[measurement[0]]['data']['timestamp'], unit='s', utc=True)},
                  index=pd.to_datetime(response.json()[measurement[0]]['data']['timestamp'], unit='s', utc=True)) if response.status_code==200 else None

stats = StatisticalTools(data_frame=df, date_col='date_time', date_format='%Y-%m-%d %H:%M', precip_col='tp')

spi = stats.compute_spi(scale=12)

period_labels = ["1991–2020", "2021-2024"]
period_reference = spi[(spi.index>pd.to_datetime('1990-12-31', utc=True)) & (spi.index<pd.to_datetime('2021-1-1', utc=True))]
period_now = spi[spi.index>pd.to_datetime('2020-12-31', utc=True)]

# Compute percentiles

ref_sorted = np.sort(period_reference.values)

n = ref_sorted.size

x = period_now.to_list()

ranks = np.searchsorted(ref_sorted, x, side="right")

percentiles = ranks / n

out = pd.DataFrame(
    {
        "value": x,
        "percentile": percentiles,
    },
    index=period_now.index,
)

out["drought_class"] = [classify_drought_from_percentile(p) for p in out["percentile"]]

def classify_drought_from_percentile(p):
    """
    Drought classes based on percentile relative to reference distribution.
    Thresholds are chosen to approximate the usual SPI/SPEI categories.
    """
    if np.isnan(p):
        return np.nan

    if p <= 0.02:
        return "Extreme drought"
    elif p <= 0.05:
        return "Severe drought"
    elif p <= 0.10:
        return "Moderate drought"
    elif p < 0.90:
        return "Near normal"
    elif p < 0.95:
        return "Moderately wet"
    elif p < 0.98:
        return "Very wet"
    else:
        return "Extremely wet"
    
print()
