import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
weights = {
    "Drought": 0.188,
    "Heatwave": 0.381,
    "Heavy Rainfall (Floods)": 0.299,
    "Erosion": 0.132
}

categories = list(weights.keys())
values = list(weights.values())
N = len(categories)

# Close the loop
values += values[:1]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# --- Polygon and fill ---
ax.plot(angles, values, linewidth=2.5, color='royalblue')
ax.fill(angles, values, color='skyblue', alpha=0.35)

# --- Custom category labels ---
label_offset = 0.1  # distance from center
for angle, label in zip(angles[:-1], categories):
    # Convert radians to degrees
    deg = np.degrees(angle)

    # Decide rotation and alignment based on angle
    if np.isclose(deg, 90):  # right
        rotation = 0
        ha, va = 'left', 'center'
    elif np.isclose(deg, 270):  # left
        rotation = 0
        ha, va = 'right', 'center'
    elif np.isclose(deg, 0) or np.isclose(deg, 360):  # top
        rotation = 0
        ha, va = 'center', 'bottom'
    elif np.isclose(deg, 180):  # bottom
        rotation = 0
        ha, va = 'center', 'top'
    else:
        rotation = np.degrees(angle) - 90
        ha, va = 'center', 'center'

    ax.text(
        angle, label_offset, label,
        size=12, fontweight='bold',
        rotation=rotation, rotation_mode='anchor',
        ha=ha, va=va
    )

    # --- Numeric labels ---
    for a, v in zip(angles[:-1], values[:-1]):
        ax.text(a, v + 0.3 * 0.17, f"{v:.2f}", ha="center", va="center", fontsize=12, color="royalblue")

# # --- Value annotations (above polygon) ---
# for angle, val in zip(angles[:-1], values[:-1]):
#     ax.text(angle, val + 0.035, f"{val:.3f}", size=11, ha='center', va='center')

# --- Radial ticks and limits ---
ax.set_rlabel_position(0)
plt.yticks([0.15, 0.3, 0.45, 0.6], ["0.15", "0.30", "0.45", "0.60"], fontsize=10, color="gray")
ax.set_ylim(0, 0.6)

# Remove default theta grid labels (we replaced them)
ax.set_xticklabels([])

# --- Title ---
# ax.set_title("Hazard Significance Distribution According to AHP Analysis",
#              pad=35, fontsize=13, fontweight='bold')

plt.tight_layout(pad=3)
plt.show()
