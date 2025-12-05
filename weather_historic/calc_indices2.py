import requests
import numpy as np
from datetime import datetime
import pandas as pd

# Calculate Relative Humidity from ERA5 single level

base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

longitude = 24.40
latitude = 40.93
start_timestamp = 0
end_timestamp =  1735680000.0

# Gettin Air Temperature data
measurement = "t2m"

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
data = pd.DataFrame(data={"t2m":[round(x-273.15,1) for x in response.json()['data']['value']]},
                    index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None

# Getting Dew Point data
measurement = "d2m"

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

data["d2m"] = [round(x-273.15,1) for x in response.json()['data']['value']]

# Define Relative Humidity as a Helper Function
def relative_humidity(T, Td):
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    RH = round(100 * (e / es), 2)
    return RH

data['rh'] = relative_humidity(data["t2m"], data["d2m"])

#  Temperature Humidity Index, THI

data["THI"] = data.apply(lambda r: round(r['t2m'] - (0.55 - 0.0055 * r['rh']) * (r['t2m'] - 14.5),1), axis=1)

# Relative Strain Index

data["water_vapor_pressure"] = data.apply(lambda r: (r['rh']/100) * 6.112 * 10**(7.5*r['t2m']/(237.7+r['t2m'])), axis=1)

data['RSI'] = data.apply(lambda r: (r['t2m'] - 21) / (58 - r['water_vapor_pressure']), axis=1)
print(data)
print()