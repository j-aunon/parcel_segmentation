"""
build_dataset.py — semantic segmentation dataset from PNOA + SIGPAC (Lleida)

Output: data/dataset/{images,masks}/{train,val,test}/*.png
        data/dataset/classes.txt   ← SIGPAC codes actually present in the data

Spatial split: territory divided into 10 km blocks, randomly assigned to
train/val/test. Patches (256 m, 1024 px) sampled at random within blocks.
Year alignment: PNOA acquisition year must match available SIGPAC year.
"""

import os, time, random
import requests, numpy as np, geopandas as gpd
from PIL import Image
from io import BytesIO
from shapely.geometry import box, Point
from rasterio.features import rasterize
from rasterio.transform import from_bounds

PATCH_M  = 256
IMG_PX   = 1024
BLOCK_M  = 10_000
N_SAMPLES = 1000
SPLITS   = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED     = 42
SIGPAC_BASE = "data/sigpac"
OUT_DIR     = "data/dataset"
WMS_URL     = "https://www.ign.es/wms-inspire/pnoa-ma"

CLASSES = {
    "TA": 1, "TH": 1, "VI": 2, "OV": 3, "FO": 4,
    "PA": 5, "PR": 5, "PS": 5, "FS": 6, "FY": 6, "CI": 6,
    "ED": 7, "ZU": 7, "AG": 8, "CA": 9, "IM": 10, "EP": 11, "IV": 12,
}


def query_pnoa_year(cx, cy):
    r = requests.get(WMS_URL, params={
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
        "LAYERS": "OI.MosaicElement", "QUERY_LAYERS": "OI.MosaicElement",
        "STYLES": "", "CRS": "EPSG:25831",
        "BBOX": f"{cx},{cy},{cx+256},{cy+256}",
        "WIDTH": 256, "HEIGHT": 256, "I": 128, "J": 128,
        "INFO_FORMAT": "text/plain",
    }, timeout=15)
    for line in r.text.splitlines():
        if "FECHA" in line:
            return int((line.split("'")[1] if "'" in line else line.split("=")[1].strip())[:4])
    return None


def download_pnoa(bbox):
    xmin, ymin, xmax, ymax = bbox
    r = requests.get(WMS_URL, params={
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": "OI.OrthoimageCoverage", "STYLES": "", "CRS": "EPSG:25831",
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": IMG_PX, "HEIGHT": IMG_PX, "FORMAT": "image/png",
    }, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def make_mask(gdf, use_col, bbox, seen_codes):
    patch_box = box(*bbox)
    shapes = []
    for _, row in gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].iterrows():
        code = str(row[use_col]).strip().upper()
        if code not in CLASSES: continue
        geom = row.geometry.intersection(patch_box)
        if not geom.is_empty:
            shapes.append((geom, CLASSES[code]))
            seen_codes.add(code)
    if not shapes: return None
    return rasterize(shapes, out_shape=(IMG_PX, IMG_PX),
                     transform=from_bounds(*bbox, IMG_PX, IMG_PX), fill=0, dtype=np.uint8)


def make_block_map(bounds, rng):
    xmin, ymin, xmax, ymax = bounds
    blocks = [(x, y) for x in np.arange(xmin, xmax, BLOCK_M)
                      for y in np.arange(ymin, ymax, BLOCK_M)]
    rng.shuffle(blocks)
    n, cut1 = len(blocks), int(len(blocks) * SPLITS["train"])
    cut2 = cut1 + int(n * SPLITS["val"])
    return {(float(bx), float(by)): ("train" if i < cut1 else "val" if i < cut2 else "test")
            for i, (bx, by) in enumerate(blocks)}


if __name__ == "__main__":
    rng = random.Random(SEED)
    totals, seen_codes = {"train": 0, "val": 0, "test": 0}, set()
    margin = PATCH_M / 2

    for comarca in sorted(os.listdir(os.path.join(SIGPAC_BASE, "2024"))):
        sigpac = {}
        for year in (2024, 2025):
            d = os.path.join(SIGPAC_BASE, str(year), comarca)
            shps = [f for f in os.listdir(d) if f.endswith(".shp")] if os.path.isdir(d) else []
            if shps:
                gdf = gpd.read_file(os.path.join(d, shps[0]))
                sigpac[year] = (gdf, next(c for c in gdf.columns if c.upper() in ("US", "USO_SIGPAC")))
        if not sigpac: continue

        base_gdf = sigpac[min(sigpac)][0]
        union, bounds = base_gdf.union_all(), base_gdf.total_bounds
        block_map = make_block_map(bounds, rng)
        saved, attempts = 0, 0
        print(f"\n[{comarca}]  SIGPAC years: {sorted(sigpac)}")

        while saved < N_SAMPLES and attempts < N_SAMPLES * 6:
            attempts += 1
            cx = rng.uniform(bounds[0] + margin, bounds[2] - margin)
            cy = rng.uniform(bounds[1] + margin, bounds[3] - margin)
            if not union.contains(Point(cx, cy)): continue

            bx, by = (cx // BLOCK_M) * BLOCK_M, (cy // BLOCK_M) * BLOCK_M
            split = block_map.get((float(bx), float(by)))
            if not split: continue
            if not (bx + margin <= cx <= bx + BLOCK_M - margin and
                    by + margin <= cy <= by + BLOCK_M - margin): continue

            bbox = (cx - margin, cy - margin, cx + margin, cy + margin)

            try:
                pnoa_year = query_pnoa_year(cx, cy); time.sleep(0.05)
            except Exception: continue
            if pnoa_year not in sigpac: continue

            gdf, use_col = sigpac[pnoa_year]
            mask = make_mask(gdf, use_col, bbox, seen_codes)
            if mask is None: continue

            try:
                img = download_pnoa(bbox); time.sleep(0.05)
            except Exception: continue

            name = f"{comarca}_{pnoa_year}_{split}_{saved:04d}"
            for subdir, data in [("images", img), ("masks", Image.fromarray(mask, "L"))]:
                d = os.path.join(OUT_DIR, subdir, split)
                os.makedirs(d, exist_ok=True)
                data.save(os.path.join(d, f"{name}.png"))

            saved += 1; totals[split] += 1
            if saved % 50 == 0:
                print(f"  {saved}/{N_SAMPLES}  train={totals['train']} val={totals['val']} test={totals['test']}", flush=True)

        print(f"  done: {saved} patches ({attempts} attempts)")

    # Write classes actually present in the data
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        for code, cls_id in sorted({c: CLASSES[c] for c in seen_codes}.items(), key=lambda x: x[1]):
            f.write(f"{cls_id}: {code}\n")

    print(f"\nSaved to {OUT_DIR}/  —  train={totals['train']} val={totals['val']} test={totals['test']}  total={sum(totals.values())}")
    print(f"Classes found: {sorted(seen_codes)}")
