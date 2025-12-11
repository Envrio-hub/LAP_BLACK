import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import os

main_path = 'C:/Users/xylop/Documents/github_repos/LAP_BLACK/weather_historic/magiko/'

# List all .csv files
csv_files = glob(os.path.join(main_path, "*.csv"))

data_frames = [pd.read_csv(fl, sep=';') for fl in csv_files]

unified_df = pd.DataFrame()

for frame in data_frames:
    print(frame['DATE'])
    if 'Barometric Pressure (AVG) (mbar)' in frame.columns:
        frame = frame.drop(columns=['Barometric Pressure (AVG) (mbar)'])
    if 'Min. Barometric Pressure (AVG) (mbar)' in frame.columns:
        frame = frame.drop(columns=['Min. Barometric Pressure (AVG) (mbar)'])
    if 'Max. Barometric Pressure (AVG) (mbar)' in frame.columns:
        frame = frame.drop(columns=['Max. Barometric Pressure (AVG) (mbar)'])
    unified_df = pd.concat([unified_df, frame], axis=0)

unified_df.index = pd.to_datetime(unified_df['DATE'], format='%d/%m/%Y')
unified_df = unified_df.drop(columns=['DATE'])

# 2. Aggregate hourly to daily precipitation (mm/day)
df_daily = unified_df['Rain']

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
