# pip install geopandas matplotlib pandas numpy
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json, textwrap

# --- CONFIG ---
user = 'xylop'
main_path = f'C:/Users/{user}/OneDrive/Active/EnvRIo/Projects/Project29_lap/gis/'
# filenames = ['coast_line_2012_city_of_Kavala.shp','coast_line_2018_city_of_Kavala.shp'] # Coastal Zones
filenames = ['corine_2000_city_of_Kavala.shp','corine_2006_city_of_Kavala.shp','corine_2012_city_of_Kavala.shp','corine_2018_city_of_Kavala.shp'] # CORINE
INPUT_PATH = [f'{main_path}{filename}' for filename in filenames] # folder with .shp OR a single .shp file
LANDUSE_FIELD = ["code_90", "code_00","Code_12","Code_18"] # CORINE land use code
# LANDUSE_FIELD = ["CODE_5_12", "CODE_5_18"] # Coastal Zones land use code
AREA_UNIT = "ha" # "m2", "ha", or "km2"
TOP_K_FOR_RADAR = 8
TITLE = "Relative Coverage of Land Uses (EPSG:3035 • Equal-Area)"

UNIT_FACTOR = {"m2": 1.0, "ha": 1/10_000, "km2": 1/1_000_000}[AREA_UNIT.lower()]
TARGET_EPSG = 3035          # equal-area; keep or reproject to this

with open('land_use_data/corine_mapping.json') as f:
    coast_line_mapping = json.loads(f.read())

def iter_shps(path):
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".shp":
        yield p
    else:
        yield from p.rglob("*.shp")

def load_stack(path, land_field):
    frames = []
    for shp in iter_shps(path):
        gdf = gpd.read_file(shp)
        if land_field not in gdf.columns:
            raise ValueError(f"{shp.name} missing '{land_field}'. Has: {list(gdf.columns)}")
        if gdf.crs is None:
            raise ValueError(f"{shp.name} has no CRS defined.")
        if gdf.crs.to_epsg() != TARGET_EPSG:
            gdf = gdf.to_crs(TARGET_EPSG)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        frames.append(gdf[[land_field, "geometry"]])
    if not frames:
        raise FileNotFoundError("No shapefiles found.")
    return pd.concat(frames, ignore_index=True)

def summarize_area_3035(gdf, land_field):
    # planar area in m² (valid because EPSG:3035 is equal-area, meters)
    df = (
        gdf.assign(area_m2=gdf.geometry.area)
           .groupby(land_field, dropna=False)["area_m2"].sum()
           .reset_index()
           .rename(columns={land_field: "landuse"})
           .sort_values("area_m2", ascending=False, ignore_index=True)
    )
    df[f"area_{AREA_UNIT}"] = df["area_m2"] * UNIT_FACTOR
    total = df["area_m2"].sum()
    df["coverage_%"] = (df["area_m2"] / total * 100).round(3)
    return df

def radar_chart(values_dict1, values_dict2=None, values_dict3=None, values_dict4=None, labels=("Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"),
                title="", rmax=None, figsize=(7.5,7.5)):
    """
    Radar chart supporting one or two datasets.

    Parameters
    ----------
    values_dict1 : dict
        {category: value} for the first dataset
    values_dict2 : dict, optional
        {category: value} for the second dataset (same categories required)
    labels : tuple of str
        Names for the legend entries
    title : str
        Chart title
    rmax : float, optional
        Maximum radial value
    figsize : tuple
        Figure size
    """

    cats = list(values_dict1.keys())
    vals1 = list(values_dict1.values())
    N = len(cats)

    # Close the loop
    vals1 += vals1[:1]
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Optional second dataset
    if values_dict2 is not None:
        vals2 = [values_dict2[k] for k in cats]
        vals2 += vals2[:1]

    # Optional second dataset
    if values_dict3 is not None:
        vals3 = [values_dict3[k] for k in cats]
        vals3 += vals3[:1]

    # Optional second dataset
    if values_dict4 is not None:
        vals4 = [values_dict4[k] for k in cats]
        vals4 += vals4[:1]

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # --- Plot datasets ---
    ax.plot(angles, vals1, linewidth=2.5, color="royalblue", label=labels[0], marker='o')
    ax.fill(angles, vals1, color="skyblue", alpha=0.35)

    if values_dict2 is not None:
        ax.plot(angles, vals2, linewidth=2.5, color="darkorange", label=labels[1], marker='o')
        ax.fill(angles, vals2, color="orange", alpha=0.25)

    if values_dict3 is not None:
        ax.plot(angles, vals3, linewidth=2.5, color="seagreen", label=labels[2], marker='o')
        ax.fill(angles, vals3, color="mediumseagreen", alpha=0.25)

    if values_dict4 is not None:
        ax.plot(angles, vals4, linewidth=2.5, color="crimson", label=labels[3], marker='o')
        ax.fill(angles, vals4, color="lightcoral", alpha=0.25)

    # --- Axis limits ---
    vmax1 = max(vals1[:-1])
    vmax2 = max(vals2[:-1]) if values_dict2 is not None else 0
    vmax3 = max(vals3[:-1]) if values_dict3 is not None else 0
    vmax4 = max(vals4[:-1]) if values_dict4 is not None else 0
    vmax = max(vmax1, vmax2, vmax3, vmax4)
    rmax = (vmax * 1.15) if rmax is None else rmax
    ax.set_ylim(0, rmax)

    # # --- Numeric labels ---
    # for a, v in zip(angles[:-1], vals1[:-1]):
    #     ax.text(a, v + rmax * 0.1, f"{v:.2f}", ha="center", va="center", fontsize=9, color="royalblue")

    # if values_dict2 is not None:
    #     for a, v in zip(angles[:-1], vals2[:-1]):
    #         ax.text(a, v + rmax * 0.23, f"{v:.2f}", ha="center", va="center", fontsize=9, color="darkorange")

    # if values_dict3 is not None:
    #     for a, v in zip(angles[:-1], vals3[:-1]):
    #         ax.text(a, v + rmax * 0.23, f"{v:.2f}", ha="center", va="center", fontsize=9, color="seagreen")

    # if values_dict4 is not None:
    #     for a, v in zip(angles[:-1], vals4[:-1]):
    #         ax.text(a, v + rmax * 0.23, f"{v:.2f}", ha="center", va="center", fontsize=9, color="crimson")

    # --- Category labels (wrapped & centered) ---
    label_margin = 0.25 * rmax
    label_r = rmax + label_margin
    ax.set_xticklabels([])

    for a, lab in zip(angles[:-1], cats):
        wrapped_lines = textwrap.wrap(lab, width=14)
        wrapped_lab = "\n".join(line.center(14) for line in wrapped_lines)
        deg = (np.degrees(a) + 360) % 360
        print('\ndeg',deg)

        # Cardinal orientation rules
        if np.isclose(deg, 0) or np.isclose(deg, 360):
            rot, ha, va = 0, "center", "center"
        elif np.isclose(deg, 180):
            rot, ha, va = 0, "center", "center"
        elif np.isclose(deg, 40):
            rot, ha, va = 0, "left", "center"
        elif np.isclose(deg, 80):
            rot, ha, va = 0, "left", "center"
        elif np.isclose(deg, 120):
            rot, ha, va = 0, "left", "center"
        elif np.isclose(deg, 160):
            rot, ha, va = 0, "left", "center"
        elif np.isclose(deg, 200):
            rot, ha, va = 0, "right", "center"
        elif np.isclose(deg, 240):
            rot, ha, va = 0, "right", "center"
        elif np.isclose(deg, 280):
            rot, ha, va = 0, "right", "center"
        elif np.isclose(deg, 320):
            rot, ha, va = 0, "right", "center"

        ax.text(a, label_r, wrapped_lab,
                rotation=rot, rotation_mode="anchor",
                ha=ha, va=va, fontsize=10, fontweight="bold",
                linespacing=1.1, clip_on=False)

    # --- Radial ticks, legend, title ---
    ax.set_rlabel_position(0)
    yt = np.linspace(0, rmax, 4)[1:]
    plt.yticks([round(t, 2) for t in yt], [f"{t:.2f}" for t in yt], fontsize=9)
    if values_dict2 is not None:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.2))
    if title:
        ax.set_title(title, pad=28, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig, ax

# --- Run ---
df_list = []
for path, landuse in zip(INPUT_PATH, LANDUSE_FIELD):
    gdf = load_stack(path, landuse)      # ensures EPSG:3035
    summary = summarize_area_3035(gdf, landuse)
    summary[f"area_{AREA_UNIT}"] = round(summary[f"area_{AREA_UNIT}"],2)

    # table preview
    print(summary[[ "landuse", f"area_{AREA_UNIT}", "coverage_%" ]].to_string(index=False))

    # optional CSV
    # summary.to_csv("landuse_area_summary_3035.csv", index=False)

    df_list.append(summary[["landuse", "area_ha", "coverage_%"]])

def _map_desc(code):
    s = str(code)
    return coast_line_mapping.get(s, {}).get("description", s)

def prepare_df_for_radar(df, top_k=8):
    """Return a tidy df with columns ['landuse','coverage_%'] where landuse is the *description*.
       Collapses everything beyond top_k into 'Other' and maps codes -> descriptions."""
    df = df.copy()
    # collapse to top-k + Other (only if needed)
    if len(df) > top_k:
        top = df.nlargest(top_k, "coverage_%").copy()
        other_pct = df["coverage_%"].sum() - top["coverage_%"].sum()
        if other_pct > 0:
            top.loc[len(top)] = {"landuse": "Other", "coverage_%": other_pct}
        df = top

    # map codes -> descriptions (keep 'Other' as-is)
    df["landuse"] = df["landuse"].astype(str)
    mask_other = df["landuse"].eq("Other")
    df.loc[~mask_other, "landuse"] = df.loc[~mask_other, "landuse"].map(_map_desc)

    # stable alphabetical order (or change to a custom order if you prefer)
    df = df.sort_values("landuse").reset_index(drop=True)
    return df[["landuse", "coverage_%"]]

# ---- merge dataframes ----
df_merged = pd.DataFrame()
for df in df_list:
    classes = [coast_line_mapping[str(int(i[1]['landuse']))]['description'] for i in df.iterrows()]
    df_new = df.drop("landuse", axis=1)
    df_new.index = classes
    df_merged = pd.concat([df_merged, df_new], axis=1)

# ---- process all dataframes ----
df1_p = prepare_df_for_radar(df_list[0], TOP_K_FOR_RADAR)
df2_p = prepare_df_for_radar(df_list[1], TOP_K_FOR_RADAR)
df3_p = prepare_df_for_radar(df_list[2], TOP_K_FOR_RADAR) if len(df_list) > 2 else None
df4_p = prepare_df_for_radar(df_list[3], TOP_K_FOR_RADAR) if len(df_list) > 3 else None

# ---- align categories across the two datasets (fill missing with 0) ----
all_cats = sorted(set(df1_p["landuse"]).union(df2_p["landuse"]).union(df3_p["landuse"]).union(df4_p['landuse']))

s1 = df1_p.set_index("landuse")["coverage_%"].reindex(all_cats, fill_value=0)
s2 = df2_p.set_index("landuse")["coverage_%"].reindex(all_cats, fill_value=0)
s3 = df3_p.set_index("landuse")["coverage_%"].reindex(all_cats, fill_value=0) if "df3_p" in locals() and df3_p is not None else None
s4 = df4_p.set_index("landuse")["coverage_%"].reindex(all_cats, fill_value=0) if "df4_p" in locals() and df4_p is not None else None

values1 = s1.to_dict()
values2 = s2.to_dict()
values3 = s3.to_dict() if s3 is not None else None
values4 = s4.to_dict() if s4 is not None else None

# ---- plot both on the same radar ----
radar_chart(values1, values2, values3, values4, labels=("2000", "2006", "2012", "2018"))
plt.show()
