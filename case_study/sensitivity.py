"""Sensitivity analyses for the Chattogram case study (run after the main run).

Uses LSM Mapper's engine on the main run's samples and aligned factors:
  S1  factor set: all 13 vs VIF-screened (VIF < 5) vs without Rainfall and Soil
  S2  rainfall layer: de-blocked (main) vs raw 30 m export (blocky)
  S3  spatial CV block size: 5, 10, 20 km
  S4  default vs tuned hyper-parameters (CV, same folds)
  M   XGBoost/RF maps for S1-S2 variants (to inspect block artefacts, R2 #9)

    python case_study/sensitivity.py MAIN_RUN_DIR RAW_RAINFALL_TIF OUT_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsm_mapper.core import raster, validation, classify  # noqa: E402
from lsm_mapper.core.models import make_model  # noqa: E402

main_dir, raw_rain, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rep = json.load(open(os.path.join(main_dir, "run_report.json")))
df = pd.read_csv(os.path.join(main_dir, "tables", "samples.csv"))
cfg = rep["config"]
seed = cfg["seed"]
cats = ["LULC", "Soil"]
all_f = [c for c in df.columns if c not in ("x", "y", "label")]
tuned = {m: rep["methods"][m]["params"] for m in ("XGBoost", "Random Forest", "Logistic Regression")}

# raw (blocky) rainfall at the sample cells, on the same grid
grid, _ = raster.reference_grid(cfg["dem"])
raw_al = os.path.join(out, "Rainfall_raw_aligned.tif")
if not os.path.exists(raw_al):
    raster.align_to_grid(raw_rain, grid, raw_al)
with rasterio.open(raw_al) as s:
    rr = np.array([v[0] for v in s.sample(df[["x", "y"]].to_numpy())])
df["Rainfall_raw"] = rr

area = pd.read_pickle(os.path.join(out, "area.pkl")) if os.path.exists(os.path.join(out, "area.pkl")) else None
if area is None:
    vm = raster.read_band(os.path.join(main_dir, "_work", "valid_mask.tif")) > 0
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(vm), 300_000, replace=False)
    r, c = np.unravel_index(idx, vm.shape)
    area = pd.DataFrame({k: raster.read_band(os.path.join(main_dir, "factors", f"{k}.tif"))[r, c] for k in all_f})
    area["Rainfall_raw"] = raster.read_band(raw_al)[r, c]
    area.to_pickle(os.path.join(out, "area.pkl"))

vif_keep, vif_drop = validation.vif_screen(area[all_f].sample(5000, random_state=seed), 5.0)
sets = {
    "All 13 factors": all_f,
    f"VIF-screened ({len(vif_keep)} factors)": vif_keep,
    "Without Rainfall and Soil": [f for f in all_f if f not in ("Rainfall", "Soil")],
    "Raw (blocky) rainfall": [f if f != "Rainfall" else "Rainfall_raw" for f in all_f],
}
methods = ["XGBoost", "Random Forest", "Logistic Regression", "Frequency Ratio"]
y = df.label.to_numpy()
xy = df[["x", "y"]].to_numpy()
rows = []


def cv(m, feats, params, scheme, block_km=10, repeats=5):
    X = df[feats].to_numpy(float)
    c = [f for f in feats if f in cats]
    a = area.rename(columns={}) if True else None
    mk = lambda: make_model(m, feats, c, params, area_sample=area, seed=seed)
    d = validation.cross_validate(mk, m, X, y, scheme=scheme, k=5, repeats=repeats,
                                  block_m=block_km * 1000, xy=xy, seed=seed)
    return d.AUC.mean(), d.AUC.std(ddof=1)


for name, feats in sets.items():
    for m in methods:
        for scheme in ("kfold", "spatial"):
            mu, sd = cv(m, feats, tuned.get(m), scheme)
            rows.append(dict(analysis="factor set", variant=name, method=m, scheme=scheme, auc=mu, sd=sd))
            print(rows[-1], flush=True)
for bk in (5, 10, 20):
    for m in methods:
        mu, sd = cv(m, all_f, tuned.get(m), "spatial", block_km=bk)
        rows.append(dict(analysis="block size", variant=f"{bk} km", method=m, scheme="spatial", auc=mu, sd=sd))
        print(rows[-1], flush=True)
for m in ("XGBoost", "Random Forest", "Logistic Regression"):
    for lab, p in (("default", None), ("tuned", tuned[m])):
        for scheme in ("kfold", "spatial"):
            mu, sd = cv(m, all_f, p, scheme)
            rows.append(dict(analysis="parameters", variant=lab, method=m, scheme=scheme, auc=mu, sd=sd))
            print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out, "sensitivity.csv"), index=False)
json.dump({"vif_kept": vif_keep, "vif_dropped": vif_drop}, open(os.path.join(out, "vif_screen.json"), "w"), indent=1)

# ---- maps for the artefact check (XGBoost and RF, three rainfall/soil variants)
if len(sys.argv) > 4 and sys.argv[4] == "maps":
    from lsm_mapper.core.validation import holdout
    tr, te = holdout(None, y, cfg["test_size"], seed)
    vm = raster.read_band(os.path.join(main_dir, "_work", "valid_mask.tif")) > 0
    paths = {k: os.path.join(main_dir, "factors", f"{k}.tif") for k in all_f}
    paths["Rainfall_raw"] = raw_al
    summary = {}
    for vname, feats in (("raw_rain", sets["Raw (blocky) rainfall"]),
                         ("no_rain_soil", sets["Without Rainfall and Soil"])):
        for m in ("XGBoost", "Random Forest"):
            c = [f for f in feats if f in cats]
            pipe = make_model(m, feats, c, tuned[m], area_sample=area, seed=seed)
            pipe.fit(df[feats].to_numpy(float)[tr], y[tr])
            prob = np.full(grid.shape, np.nan, np.float32)
            srcs = {k: rasterio.open(paths[k]) for k in feats}
            for r0 in range(0, grid.height, 256):
                h = min(256, grid.height - r0)
                w = rasterio.windows.Window(0, r0, grid.width, h)
                vmb = vm[r0:r0 + h]
                if not vmb.any():
                    continue
                Xb = np.stack([raster.clean_nodata(srcs[k].read(1, window=w).astype(np.float32),
                                                   srcs[k].nodata)[vmb] for k in feats], 1)
                ok = np.isfinite(Xb).all(1)
                blk = prob[r0:r0 + h]
                tmp = np.full(ok.shape, np.nan, np.float32)
                tmp[ok] = pipe.predict_proba(Xb[ok])[:, 1]
                blk[vmb] = tmp
            for s in srcs.values():
                s.close()
            fn = os.path.join(out, f"LSM_{m.replace(' ', '_')}_{vname}_probability.tif")
            raster.write_array(prob, grid, fn)
            from sklearn.metrics import roc_auc_score
            p_te = pipe.predict_proba(df[feats].to_numpy(float)[te])[:, 1]
            summary[f"{m}|{vname}"] = {"test_auc": float(roc_auc_score(y[te], p_te))}
            print(m, vname, summary[f"{m}|{vname}"], flush=True)
    json.dump(summary, open(os.path.join(out, "variant_maps.json"), "w"), indent=1)
