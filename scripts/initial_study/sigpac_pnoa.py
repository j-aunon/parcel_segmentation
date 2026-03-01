import os
import numpy as np
import requests
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# --- CONFIG ---
SIGPAC_SHP = "/home/jaunon/Projects/parcel_segmentation/data/sigpac/2025/Segria/SIGPAC_33_Segria.shp"

BBOX = (295000, 4610000, 295256, 4610256)  # EPSG:25831, meters — 256m x 256m test
IMG_SIZE = 1024                            # 0.25 m/pixel (same GSD as PNOA/paper)
OUT_DIR = "dataset_lleida"

CLASSES = {
    "TA": (1,  "#FFD700"), "TH": (1,  "#FFD700"),  # arable land
    "VI": (2,  "#800080"),                           # vineyard
    "OV": (3,  "#8B4513"),                           # olive grove
    "FO": (4,  "#228B22"),                           # forest
    "PA": (5,  "#90EE90"), "PR": (5,  "#90EE90"), "PS": (5, "#90EE90"),   # pasture
    "FS": (6,  "#FF8C00"), "FY": (6,  "#FF8C00"), "CI": (6, "#FF8C00"),   # fruit trees
    "ED": (7,  "#FF4500"), "ZU": (7,  "#FF4500"),   # urban
    "AG": (8,  "#1E90FF"),                           # water
    "CA": (9,  "#A9A9A9"),                           # roads
    "IM": (10, "#808080"),                           # unproductive
    "EP": (11, "#DEB887"),                           # landscape element
    "IV": (12, "#00CED1"),                           # greenhouse
}


def get_pnoa(bbox, size=IMG_SIZE):
    r = requests.get("https://www.ign.es/wms-inspire/pnoa-ma", params={
        "SERVICE": "WMS",                                    # protocol
        "VERSION": "1.3.0",                                  # WMS version
        "REQUEST": "GetMap",                                 # request an image
        "LAYERS": "OI.OrthoimageCoverage",                   # INSPIRE layer name for PNOA orthophotos
        "STYLES": "",                                        # default style
        "CRS": "EPSG:25831",                                 # UTM zone 31N (metric system used in Spain)
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",  # xmin,ymin,xmax,ymax in meters
        "WIDTH": size, "HEIGHT": size,                       # output image size in pixels
        "FORMAT": "image/png",                               # output format
    }, timeout=60)
    r.raise_for_status()
    return np.array(Image.open(BytesIO(r.content)).convert("RGB"))


def load_sigpac(shp_path, bbox):
    """Load SIGPAC from local SHP, filtering to bbox to avoid loading entire comarca."""
    from shapely.geometry import box
    print(f"  Loading {shp_path}...")
    gdf = gpd.read_file(shp_path, bbox=box(*bbox))  # spatial filter at read time
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")

    # Ensure EPSG:25831
    if gdf.crs.to_epsg() != 25831:
        gdf = gdf.to_crs("EPSG:25831")

    return gdf


def make_mask(gdf, bbox, size=IMG_SIZE):
    use_col = next((c for c in gdf.columns if c.upper() in ("US", "USO_SIGPAC")), None)
    if use_col is None:
        raise ValueError(f"Land-use column not found. Available: {list(gdf.columns)}")

    print(f"  Use column: '{use_col}'")
    print(gdf[use_col].value_counts().to_string())

    shapes = [
        (geom, CLASSES[code][0])
        for geom, raw in zip(gdf.geometry, gdf[use_col])
        if (code := str(raw).strip().upper()) in CLASSES
    ]
    print(f"  Matched shapes: {len(shapes)}")

    return rasterize(shapes, out_shape=(size, size),
                     transform=from_bounds(*bbox, size, size),
                     fill=0, dtype=np.uint8)


def visualize(img, mask, bbox):
    id_to_color = {cls_id: col for cls_id, col in CLASSES.values()}
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in id_to_color.items():
        mask_rgb[mask == cls_id] = (np.array(to_rgb(color)) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(img);         axes[0].set_title("PNOA")
    axes[1].imshow(mask_rgb);    axes[1].set_title("SIGPAC mask")
    axes[2].imshow(img);         axes[2].imshow(mask_rgb, alpha=0.45)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"SIGPAC + PNOA — Lleida | bbox: {bbox}", fontsize=11, y=1.01)
    plt.tight_layout(pad=1.0)
    out = f"{OUT_DIR}/result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading PNOA...")
    img = get_pnoa(BBOX)
    print(f"  shape: {img.shape}")

    print("Loading SIGPAC...")
    gdf = load_sigpac(SIGPAC_SHP, BBOX)
    print(f"  {len(gdf)} parcels in bbox")

    print("Creating mask...")
    mask = make_mask(gdf, BBOX)
    print(f"  {100*(mask > 0).mean():.1f}% annotated | classes: {np.unique(mask)}")

    print("Visualizing...")
    visualize(img, mask, BBOX)