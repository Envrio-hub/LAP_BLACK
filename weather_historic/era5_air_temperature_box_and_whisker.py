import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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
start = datetime.now()
response = requests.get(f'{base_url}/time_series_data', headers=headers, params=params)

data = pd.DataFrame(data={"values":[round(x-273.15,1) for x in response.json()['data']['value']]},
                    index=[datetime.fromtimestamp(dt) for dt in response.json()['data']['timestamp']]) if response.status_code==200 else None
print(f'Duration: {datetime.now()-start}')
data['year'] = data.index.year

# Define seasons using month mapping
season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}
data['season'] = data.index.month.map(season_map)

# Group by year and season
grouped = data.groupby(['year', 'season'])['values']

import matplotlib.pyplot as plt

seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
years = sorted(data['year'].unique())

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
axes = axes.ravel()  # flatten: 0..3

for ax, season in zip(axes, seasons):

    # Filter available years for this season
    available = [yr for yr in years if (yr, season) in grouped.groups]

    # Build labels and season_data together to ensure same length/order
    labels = []
    season_data = []
    for yr in available:
        labels.append(yr)
        season_data.append(grouped.get_group((yr, season)).values)

    # Now lengths always match
    ax.boxplot(
        season_data,
        labels=labels,
        showmeans=True,
        medianprops={'color': 'red', 'linewidth': 1.5}
    )

    ax.set_title(season, fontweight='bold')
    ax.tick_params(axis='x', rotation=90)

fig.tight_layout()
fig.supylabel('Air Temperature (\u00B0C)',
              x=0.01,
              fontsize=14,
              fontweight='bold')

fig.subplots_adjust(
    left=0.05,
    right=0.97,
    top=0.93,
    bottom=0.08,
    wspace=0.12,
    hspace=0.22
)
plt.show()

print(variables.json())