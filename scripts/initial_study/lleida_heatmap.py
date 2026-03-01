import os, time
import requests
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Point

WMS_URL    = "https://www.ign.es/wms-inspire/pnoa-ma"
SIGPAC_DIR = "data/sigpac/2024"
OUT_PATH   = "heatmap_lleida.png"
STEP       = 4000  # meters between sample points


def query_pnoa_date(x, y):
    # Queries OI.MosaicElement layer to get acquisition date (YYYY-MM) at point (x, y)
    r = requests.get(WMS_URL, params={
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
        "LAYERS": "OI.MosaicElement", "QUERY_LAYERS": "OI.MosaicElement",
        "STYLES": "", "CRS": "EPSG:25831",
        "BBOX": f"{x},{y},{x+256},{y+256}",
        "WIDTH": 256, "HEIGHT": 256, "I": 128, "J": 128,
        "INFO_FORMAT": "text/plain",
    }, timeout=15)
    for line in r.text.splitlines():
        if "FECHA" in line:
            return line.split("'")[1] if "'" in line else line.split("=")[1].strip()
    return None


def classify(fecha):
    # Returns alignment score: 2=green (feb-may)  1=yellow (jan/jun)  -1=red (jul+)
    if fecha is None: return -1
    m = int(fecha[5:])
    return 2 if 2 <= m <= 5 else 1 if m in (1, 6) else -1


def sample_comarca(gdf):
    # Samples a grid of points inside the comarca boundary and classifies each one
    union = gdf.union_all()
    b = gdf.total_bounds
    cols = list(np.arange(b[0], b[2], STEP))
    xs, ys, cs = [], [], []
    for i, x in enumerate(cols):
        for y in np.arange(b[1], b[3], STEP):
            if not union.contains(Point(x, y)): continue
            xs.append(x); ys.append(y)
            cs.append(classify(query_pnoa_date(x, y)))
            time.sleep(0.04)  # avoid hammering the IGN server
        print(f"  col {i+1}/{len(cols)}", end="\r", flush=True)
    return xs, ys, cs


if __name__ == "__main__":
    all_xs, all_ys, all_cs, all_gdfs = [], [], [], []

    # --- Sample all comarques ---
    for comarca in sorted(os.listdir(SIGPAC_DIR)):
        shps = [f for f in os.listdir(f"{SIGPAC_DIR}/{comarca}") if f.endswith(".shp")]
        if not shps: continue
        print(f"Sampling {comarca}...")
        gdf = gpd.read_file(f"{SIGPAC_DIR}/{comarca}/{shps[0]}")
        xs, ys, cs = sample_comarca(gdf)
        all_xs.extend(xs); all_ys.extend(ys); all_cs.extend(cs); all_gdfs.append(gdf)
        n = len(xs)
        print(f"  {comarca}: {n} pts | "
              f"same={100*cs.count(2)/n:.0f}% "
              f"+-1m={100*(cs.count(2)+cs.count(1))/n:.0f}% "
              f"far={100*cs.count(-1)/n:.0f}%")

    # --- Plot ---
    n = len(all_cs)
    fig, ax = plt.subplots(figsize=(12, 14))
    pd.concat(all_gdfs).dissolve().boundary.plot(ax=ax, color="#333333", linewidth=0.8)
    for gdf in all_gdfs: gdf.dissolve().boundary.plot(ax=ax, color="#888888", linewidth=0.4)
    ax.scatter(all_xs, all_ys, c=["#2ca02c" if c==2 else "#ffdd57" if c==1 else "#d62728" for c in all_cs], s=18, zorder=3, edgecolors="none")
    ax.set_title(f"PNOA–SIGPAC 2024 temporal alignment — Lleida\n"
                 f"Same period: {100*all_cs.count(2)/n:.0f}%  |  ±1 month: {100*(all_cs.count(2)+all_cs.count(1))/n:.0f}%  |  Far: {100*all_cs.count(-1)/n:.0f}%",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in [
        ("#2ca02c", "Same period (feb–may)"), ("#ffdd57", "±1 month"), ("#d62728", "Far (jul+)"),
    ]], loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {OUT_PATH}")
