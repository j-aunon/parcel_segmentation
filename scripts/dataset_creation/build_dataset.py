import os, time, random, csv
import requests, numpy as np, geopandas as gpd
from PIL import Image
from io import BytesIO
from shapely.geometry import box, Point
from rasterio.features import rasterize
from rasterio.transform import from_bounds

PATCH_M     = 256
IMG_PX      = 1024
BLOCK_M     = 10_000
N_SAMPLES   = 1
SEED        = 42
SIGPAC_BASE = "data/sigpac"
OUT_DIR     = "data/dataset"
WMS_URL     = "https://www.ign.es/wms-inspire/pnoa-ma"
WMS_BASE    = {"SERVICE": "WMS", "VERSION": "1.3.0", "STYLES": "", "CRS": "EPSG:25831"}
SPLITS      = {"train": 0.70, "val": 0.15, "test": 0.15}


def get_pnoa_year(cx, cy):
    r = requests.get(WMS_URL, params={**WMS_BASE, "REQUEST": "GetFeatureInfo",
        "LAYERS": "OI.MosaicElement", "QUERY_LAYERS": "OI.MosaicElement",
        "BBOX": f"{cx},{cy},{cx+256},{cy+256}", "WIDTH": 256, "HEIGHT": 256,
        "I": 128, "J": 128, "INFO_FORMAT": "text/plain"}, timeout=15)
    for line in r.text.splitlines():
        if "FECHA" in line:
            val = line.split("'")[1] if "'" in line else line.split("=")[1].strip()
            return int(val[:4])


def get_pnoa_image(bbox):
    xmin, ymin, xmax, ymax = bbox
    r = requests.get(WMS_URL, params={**WMS_BASE, "REQUEST": "GetMap",
        "LAYERS": "OI.OrthoimageCoverage", "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": IMG_PX, "HEIGHT": IMG_PX, "FORMAT": "image/png"}, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def make_mask(gdf, col, bbox, classes):
    shapes = []
    for _, row in gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].iterrows():
        code = str(row[col]).strip().upper()
        if not code or code == "NAN": continue
        geom = row.geometry.intersection(box(*bbox))
        if geom.is_empty: continue
        classes.setdefault(code, len(classes) + 1)
        shapes.append((geom, classes[code]))
    if not shapes: return None
    return rasterize(shapes, out_shape=(IMG_PX, IMG_PX),
                     transform=from_bounds(*bbox, IMG_PX, IMG_PX), fill=0, dtype=np.uint8)


def make_split_map(bounds, rng):
    x0, y0 = (bounds[0] // BLOCK_M) * BLOCK_M, (bounds[1] // BLOCK_M) * BLOCK_M
    blocks = [(float(x), float(y)) for x in np.arange(x0, bounds[2], BLOCK_M)
                                    for y in np.arange(y0, bounds[3], BLOCK_M)]
    rng.shuffle(blocks)
    n, c1, c2 = len(blocks), int(len(blocks) * SPLITS["train"]), 0
    c2 = c1 + int(n * SPLITS["val"])
    return {b: ("train" if i < c1 else "val" if i < c2 else "test")
            for i, b in enumerate(blocks)}


if __name__ == "__main__":
    rng = random.Random(SEED)
    classes, records = {}, []
    half = PATCH_M / 2
    os.makedirs(OUT_DIR, exist_ok=True)

    for comarca in sorted(os.listdir(os.path.join(SIGPAC_BASE, "2024"))):
        sigpac = {}
        for year in (2024, 2025):
            d = os.path.join(SIGPAC_BASE, str(year), comarca)
            shps = [f for f in os.listdir(d) if f.endswith(".shp")] if os.path.isdir(d) else []
            if shps:
                gdf = gpd.read_file(os.path.join(d, shps[0]))
                col = next(c for c in gdf.columns if c.upper() in ("US", "USO_SIGPAC"))
                sigpac[year] = (gdf, col)
        if not sigpac: continue

        base_gdf = sigpac[min(sigpac)][0]
        split_map = make_split_map(base_gdf.total_bounds, rng)
        polys = [g for g in base_gdf.geometry if g and not g.is_empty and g.is_valid]
        areas = [g.area for g in polys]
        saved = 0
        print(f"\n[{comarca}]  years={sorted(sigpac)}")

        while saved < N_SAMPLES:
            [poly] = rng.choices(polys, weights=areas, k=1)
            x0, y0, x1, y1 = poly.bounds
            for _ in range(20):
                cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if poly.contains(Point(cx, cy)): break
            else: continue

            bx, by = (cx // BLOCK_M) * BLOCK_M, (cy // BLOCK_M) * BLOCK_M
            split = split_map.get((float(bx), float(by)))
            if not split: continue
            if not (bx + half <= cx <= bx + BLOCK_M - half and
                    by + half <= cy <= by + BLOCK_M - half): continue

            bbox = (cx - half, cy - half, cx + half, cy + half)

            try:
                year = get_pnoa_year(cx, cy); time.sleep(0.05)
            except Exception: continue
            if year not in sigpac: continue

            gdf, col = sigpac[year]
            mask = make_mask(gdf, col, bbox, classes)
            if mask is None: continue

            try:
                img = get_pnoa_image(bbox); time.sleep(0.05)
            except Exception: continue

            name = f"{comarca}_{year}_{saved:04d}"
            for sub, data in [("images", img), ("masks", Image.fromarray(mask, "L"))]:
                out = os.path.join(OUT_DIR, sub, split)
                os.makedirs(out, exist_ok=True)
                data.save(os.path.join(out, f"{name}.png"))

            xmin, ymin, xmax, ymax = bbox
            records.append({"filename": f"{name}.png", "split": split,
                            "comarca": comarca, "year": year,
                            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
            saved += 1
            if saved % 50 == 0:
                print(f"  {saved}/{N_SAMPLES}", flush=True)

        print(f"  done: {saved} patches")

    # --- CSV (4 corners per patch, EPSG:25831) ---
    with open(os.path.join(OUT_DIR, "patches.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "split", "comarca", "year", "xmin", "ymin", "xmax", "ymax"])
        w.writeheader()
        w.writerows(records)

    # --- classes.txt ---
    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        f.write("0: background\n")
        for code, cid in sorted(classes.items(), key=lambda x: x[1]):
            f.write(f"{cid}: {code}\n")

    total = len(records)
    print(f"\nTotal: {total} patches → {OUT_DIR}")
    print(f"Classes: {len(classes)}  |  CSV: patches.csv")
