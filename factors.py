import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
import os

def derive_slope(dem):
    dy, dx = np.gradient(dem)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)

def derive_aspect(dem):
    dy, dx = np.gradient(dem)
    return (np.degrees(np.arctan2(-dy, dx)) % 360).astype(np.float32)

def derive_curvature(dem):
    dy, dx     = np.gradient(dem)
    ddy, _     = np.gradient(dy)
    _, ddx     = np.gradient(dx)
    return np.clip(ddy + ddx, -10, 10).astype(np.float32)

def derive_plan_curvature(dem):
    dy, dx = np.gradient(dem)
    _, ddx = np.gradient(dx)
    return np.clip(ddx, -10, 10).astype(np.float32)

def derive_profile_curvature(dem):
    dy, dx  = np.gradient(dem)
    ddy, _  = np.gradient(dy)
    return np.clip(ddy, -10, 10).astype(np.float32)

def derive_twi(dem):
    slope   = derive_slope(dem)
    slope_r = np.radians(np.where(slope < 1, 1, slope))
    flow    = np.clip(dem.max() - dem + 1, 1, None)
    return np.clip(np.log(flow / np.tan(slope_r)), 0, 20).astype(np.float32)

def derive_spi(dem):
    slope   = derive_slope(dem)
    slope_r = np.radians(np.where(slope < 1, 1, slope))
    flow    = np.clip(dem.max() - dem + 1, 1, None)
    return np.clip(np.log1p(flow * np.tan(slope_r)), 0, 15).astype(np.float32)

def shapefile_to_distance_raster(shp_path, ref_tif_path):
    """Burn shapefile to raster then compute distance in metres."""
    with rasterio.open(ref_tif_path) as src:
        meta      = src.meta.copy()
        transform = src.transform
        shape     = (src.height, src.width)
        crs       = src.crs

    gdf = gpd.read_file(shp_path).to_crs(crs)

    from rasterio.features import rasterize
    from shapely.geometry import mapping

    burned = rasterize(
        [(mapping(geom), 1) for geom in gdf.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    dist = distance_transform_edt(1 - burned)
    pixel_size = abs(transform.a)
    dist_m = (dist * pixel_size * 111320).astype(np.float32)
    return dist_m

def save_factor(array, meta, out_path):
    m = meta.copy()
    m.update(dtype="float32", count=1)
    with rasterio.open(out_path, "w", **m) as dst:
        dst.write(array.astype(np.float32), 1)

def derive_all_factors(dem_path, out_dir, optional_files=None):
    """
    Derives all factors from DEM automatically.
    optional_files: dict of name -> path for extra uploaded files.
    Returns dict of factor_name -> tif_path
    """
    os.makedirs(out_dir, exist_ok=True)
    optional_files = optional_files or {}

    with rasterio.open(dem_path) as src:
        dem  = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    # ── Always derived from DEM ─────────────────────────────────────────────
    auto_factors = {
        "Elevation":         dem,
        "Slope":             derive_slope(dem),
        "Aspect":            derive_aspect(dem),
        "Curvature":         derive_curvature(dem),
        "Plan_Curvature":    derive_plan_curvature(dem),
        "Profile_Curvature": derive_profile_curvature(dem),
        "TWI":               derive_twi(dem),
        "SPI":               derive_spi(dem),
    }

    factor_paths = {}
    for name, arr in auto_factors.items():
        path = os.path.join(out_dir, f"{name}.tif")
        save_factor(arr, meta, path)
        factor_paths[name] = path
        print(f"  Auto-derived: {name}")

    # ── Optional rasters uploaded by user ───────────────────────────────────
    raster_opts = ["Soil", "Geology", "LULC", "Rainfall", "NDVI"]
    for name in raster_opts:
        if name in optional_files:
            factor_paths[name] = optional_files[name]
            print(f"  Using uploaded: {name}")

    # ── Optional shapefiles → distance rasters ──────────────────────────────
    shp_opts = {
        "Roads":    "Distance_to_Road",
        "Drainage": "Distance_to_Drainage",
        "Faults":   "Distance_to_Fault",
    }
    for shp_name, factor_name in shp_opts.items():
        if shp_name in optional_files:
            print(f"  Computing {factor_name} from {shp_name} shapefile...")
            dist = shapefile_to_distance_raster(
                optional_files[shp_name], dem_path)
            path = os.path.join(out_dir, f"{factor_name}.tif")
            save_factor(dist, meta, path)
            factor_paths[factor_name] = path
            print(f"  Saved: {factor_name}")

    print(f"\nTotal factors ready: {len(factor_paths)}")
    return factor_paths