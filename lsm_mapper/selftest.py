"""Self-test on synthetic data: checks that the installed program (or the
Windows executable, `LSM_Mapper.exe --selftest`) can run the whole workflow."""
import os
import sys
import tempfile


def make_data(d):
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point
    rng = np.random.default_rng(3)
    H, W = 300, 260
    r, c = np.mgrid[0:H, 0:W]
    dem = (400 * np.exp(-((c - 130) ** 2) / 6000) * (1 - r / 400) + 20 * np.sin(r / 15)
           + rng.normal(0, 1, (H, W))).astype("float32")
    t = from_origin(500000, 2500000, 30, 30)
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32", crs="EPSG:32646", transform=t)
    with rasterio.open(os.path.join(d, "dem.tif"), "w", **prof) as f:
        f.write(dem, 1)
    lc = np.where(c < 90, 10, np.where(c < 170, 30, 40)).astype("uint8")
    with rasterio.open(os.path.join(d, "lulc.tif"), "w", **dict(prof, dtype="uint8", nodata=0)) as f:
        f.write(lc, 1)
    gy, gx = np.gradient(dem, 30)
    s = np.hypot(gx, gy)
    p = s / s.sum()
    idx = rng.choice(H * W, 150, replace=False, p=p.ravel())
    pts = [Point(t.c + (i % W + .5) * 30, t.f - (i // W + .5) * 30) for i in idx]
    gpd.GeoDataFrame(geometry=pts, crs="EPSG:32646").to_file(os.path.join(d, "inventory.shp"))


def main():
    from lsm_mapper.core.pipeline import Run, RunConfig
    d = tempfile.mkdtemp(prefix="lsm_selftest_")
    make_data(d)
    cfg = RunConfig(dem=os.path.join(d, "dem.tif"), inventory=os.path.join(d, "inventory.shp"),
                    output_dir=os.path.join(d, "out"),
                    factors=[dict(name="LULC", paths=[os.path.join(d, "lulc.tif")], kind="categorical")],
                    methods=["XGBoost", "Random Forest", "Logistic Regression", "Frequency Ratio"],
                    tune=True, tune_iter=4, tune_inner_folds=2, cv_folds=3, cv_repeats=1,
                    spatial_block_km=2.0, spatial_repeats=1, nested_cv=False, absence_min_slope=None)
    rep = Run(cfg, log=lambda s: print(s, flush=True)).run()
    ok = all(rep["methods"][m]["test"]["AUC"] > 0.5 for m in cfg.methods) and \
        os.path.exists(os.path.join(cfg.output_dir, "maps", "interactive_map.html"))
    print("SELFTEST", "PASSED" if ok else "FAILED", d)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
