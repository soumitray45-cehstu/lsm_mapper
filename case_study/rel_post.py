"""Post-processing of the reliability outputs for the paper.

For the main run (whole-area non-landslide sampling) and the first-version
design (slopes > 3 deg within the inventory box): how much of the near-flat
terrain, and of the High/Very High cells on it, lies outside the area of
applicability, and how uncertainty relates to dissimilarity.

    python case_study/rel_post.py MAIN_DIR SLOPE3_DIR OUT.json
"""
import json
import sys

import numpy as np
import rasterio


def read(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(np.float32)
        nd = s.nodata
    if nd is not None and not np.isnan(nd):
        a[a == nd] = np.nan
    return a


main, alt, out = sys.argv[1:4]
slope = read(f"{main}/factors/Slope.tif")
flat = np.isfinite(slope) & (slope < 3)
steep = np.isfinite(slope) & (slope >= 3)
res = {}
for tag, d in (("main", main), ("slope3", alt)):
    for m in ("XGBoost", "Random_Forest"):
        try:
            mask = read(f"{d}/maps/AOA_mask_{m}.tif")
        except Exception:
            continue
        cls = read(f"{d}/maps/LSM_{m}_classes.tif")
        sd = read(f"{d}/maps/Uncertainty_SD_{m}.tif")
        di = read(f"{d}/maps/AOA_DI_{m}.tif")
        v = mask > 0
        out_ = mask == 2
        hv = np.isin(cls, (4, 5))
        hvflat = hv & flat
        rng = np.random.default_rng(0)
        idx = rng.choice(np.flatnonzero(v & np.isfinite(sd) & np.isfinite(di)), 200_000, replace=False)
        from scipy.stats import spearmanr
        rho = spearmanr(di.ravel()[idx], sd.ravel()[idx]).correlation
        res[f"{tag}|{m}"] = {
            "outside_pct": float(100 * out_.sum() / v.sum()),
            "flat_outside_pct": float(100 * (out_ & flat).sum() / max((v & flat).sum(), 1)),
            "steep_outside_pct": float(100 * (out_ & steep).sum() / max((v & steep).sum(), 1)),
            "outside_that_is_flat_pct": float(100 * (out_ & flat).sum() / max(out_.sum(), 1)),
            "hv_on_flat_outside_pct": float(100 * (hvflat & out_).sum() / max(hvflat.sum(), 1)),
            "hv_outside_pct": float(100 * (hv & out_).sum() / max(hv.sum(), 1)),
            "sd_mean_flat": float(np.nanmean(sd[flat & v])),
            "sd_mean_steep": float(np.nanmean(sd[steep & v])),
            "sd_mean_hv_flat": float(np.nanmean(sd[hvflat])) if hvflat.any() else None,
            "sd_mean_hv_steep": float(np.nanmean(sd[hv & steep])),
            "sd_mean_inside": float(np.nanmean(sd[mask == 1])),
            "sd_mean_outside": float(np.nanmean(sd[out_])),
            "spearman_di_sd": float(rho),
        }
        print(tag, m, res[f"{tag}|{m}"], flush=True)
json.dump(res, open(out, "w"), indent=1)
