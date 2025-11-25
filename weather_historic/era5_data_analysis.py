import requests
from climate_indices import ClimateIndicesCalculator
from datetime import datetime


base_url = "https://envrio.org/era5_api"

auth = requests.post(f'{base_url}/auth', json={"username":"xylopodaros@yahoo.gr","password":"TestPass123@123"})

headers = {"Authorization": f'Bearer {auth.json()["access_token"]}'}

variables = requests.get(f'{base_url}/variables', headers=headers)

measurement = "t2m"

# info = requests.get(f'{base_url}/time_series_info', headers=headers, params={"measurements": [measurement]})

# print(info.json())

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

data = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)

print(variables.json())