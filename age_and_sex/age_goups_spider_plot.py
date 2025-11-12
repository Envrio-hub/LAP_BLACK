import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import fill

df = pd.read_csv('age_groups_population.csv', sep=',')
df = df.rename(columns={"Infants_and_children": "Infants & Children", "Young_adults": "Young Adults",
                        "Middle_aged_adults": "Middle-Aged Adults", "Mature_adults": "Mature Adults",
                        "Seniors": "Seniors"})

# Select which years to plot
years = [2015, 2020, 2025, 2029]
subset = df[df['year'].isin(years)]
subset = subset.set_index('year')

# Indicators (everything except 'year')
categories = list(subset.columns)
N = len(categories)

# ----- angles -----
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# put first category at the top and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# choose r-limits from data (+10% headroom)
rmax = 1.1 * np.nanmax(df[categories].values)
ax.set_ylim(0, rmax)

# tidy radial ticks/labels
ax.set_rlabel_position(180)
ax.tick_params(axis='y', pad=6)

# hide default category tick labels; keep ticks for grid
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])

# ====== plot the polygons ======
for y in years:
    row = df.loc[df['year'] == y, categories].mean()  # adjust to your structure
    vals = row.tolist()
    vals += vals[:1]
    ax.plot(angles, vals, label=str(y))
    ax.fill(angles, vals, alpha=0.1)

# make outer circle clearer
ax.spines['polar'].set_visible(True)
ax.spines['polar'].set_linewidth(1.2)
ax.grid(True, linewidth=0.6, alpha=0.6)

# ====== Custom category labels (your block, slightly adapted) ======
label_offset = 1.1  # as a fraction of the current rmax
label_r = rmax * label_offset

for angle, label in zip(angles[:-1], categories):
    deg = round(np.degrees(angle), 1)  # round avoids float precision issues

    # default alignment
    rotation, ha, va = 0, 'center', 'center'

    # Adjust alignment based on angle (matching your 5 categories)
    if np.isclose(deg, 0.0):            # right (0°)
        ha, va = 'center', 'top'
    elif np.isclose(deg, 72.0):         # bottom-right
        ha, va = 'left', 'center'
    elif np.isclose(deg, 144.0):        # bottom-left
        ha, va = 'left', 'center'
    elif np.isclose(deg, 216.0):        # top-left
        ha, va = 'right', 'center'
    elif np.isclose(deg, 288.0):        # top-right
        ha, va = 'right', 'center'

    # Add text label
    ax.text(
        angle, label_r, label,
        size=12, fontweight='bold',
        rotation=rotation,
        rotation_mode='anchor',
        ha=ha, va=va
    )

ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()
