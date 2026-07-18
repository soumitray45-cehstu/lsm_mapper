import numpy as np
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen

def load_landslide_points(shp_path, dem_path):
    gdf = gpd.read_file(shp_path)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_bounds = src.bounds
    gdf = gdf.to_crs(dem_crs)

    # ── CRS mismatch diagnostic ─────────────────────
    pts_in_bounds = 0
    for geom in gdf.geometry:
        if (dem_bounds.left <= geom.x <= dem_bounds.right
                and dem_bounds.bottom <= geom.y
                <= dem_bounds.top):
            pts_in_bounds += 1

    pct = 100 * pts_in_bounds / max(len(gdf), 1)
    print(f"Landslide points inside DEM extent: "
          f"{pts_in_bounds}/{len(gdf)} ({pct:.1f}%)")

    if pct < 50:
        print("WARNING: Less than 50% of landslide "
              "points fall inside DEM extent!")
        print("This likely means CRS mismatch or "
              "wrong DEM file uploaded.")
    elif pct < 80:
        print("WARNING: Some landslide points fall "
              "outside DEM extent — check your files.")
    else:
        print("OK: Landslide points align well "
              "with DEM extent.")

    coords = [(geom.x, geom.y)
              for geom in gdf.geometry]
    return coords

def sample_non_landslide(dem_path, n_samples, landslide_coords, buffer_m=500):
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        crs    = src.crs

    np.random.seed(42)
    non_slide = []
    attempts  = 0

    while len(non_slide) < n_samples and attempts < n_samples * 20:
        lon = np.random.uniform(bounds.left,   bounds.right)
        lat = np.random.uniform(bounds.bottom, bounds.top)
        too_close = any(
            abs(lon - lx) < 0.005 and abs(lat - ly) < 0.005
            for lx, ly in landslide_coords
        )
        if not too_close:
            non_slide.append((lon, lat))
        attempts += 1

    print(f"Sampled {len(non_slide)} non-landslide points.")
    return non_slide

def build_dataset(factor_paths, landslide_coords, non_slide_coords):
    import pandas as pd

    all_coords = landslide_coords + non_slide_coords
    labels     = [1] * len(landslide_coords) + [0] * len(non_slide_coords)

    records = []
    for name, path in factor_paths.items():
        with rasterio.open(path) as src:
            vals = [v[0] for v in src.sample(all_coords)]
        records.append((name, vals))

    df = pd.DataFrame({name: vals for name, vals in records})
    df["label"] = labels
    df = df.dropna()
    return df