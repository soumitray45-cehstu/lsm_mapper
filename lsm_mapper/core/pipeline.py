"""End-to-end LSM workflow used by both the GUI and the command line.

    prepare grid -> align / derive factors -> samples -> multicollinearity ->
    for each method: hold-out fit (+ optional tuning) -> test metrics ->
    k-fold and spatial CV -> importance -> map -> classes -> run report
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass, field, asdict

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from . import classify, raster, reliability, sampling, terrain, validation
from .models import ALL_METHODS, STAT_METHODS, make_model, params_of

DEM_FACTORS = ["Elevation", "Slope", "Aspect", "Curvature", "Plan_Curvature",
               "Profile_Curvature", "TWI", "SPI", "Dist_Drainage"]


@dataclass
class FactorSpec:
    name: str
    paths: list                      # one file, or several tiles to mosaic
    kind: str = "continuous"         # continuous | categorical | distance
    smooth_to_m: float | None = None # aggregate+interpolate an up-sampled layer
    resampling: str | None = None    # override automatic choice


@dataclass
class RunConfig:
    dem: str
    inventory: str
    output_dir: str
    roi: str | None = None
    factors: list = field(default_factory=list)          # FactorSpec dicts
    dem_factors: list = field(default_factory=lambda: list(DEM_FACTORS))
    stream_area_km2: float = 1.0
    target_crs: str | None = None
    target_res: float | None = None
    # sampling
    absence_ratio: float = 1.0
    absence_region: str = "study_area"
    absence_min_slope: float | None = None
    absence_buffer_m: float = 100.0
    seed: int = 42
    test_size: float = 0.3
    # modelling
    methods: list = field(default_factory=lambda: ["XGBoost", "Random Forest",
                                                   "Logistic Regression", "Frequency Ratio"])
    params: dict = field(default_factory=dict)            # method -> overrides
    search_spaces: dict = field(default_factory=dict)     # method -> {"clf__param": [values]}
    exclude_factors: list = field(default_factory=list)
    vif_screen: float | None = None                       # e.g. 5.0 to drop collinear factors
    tune: bool = True
    tune_search: str = "random"
    tune_iter: int = 40
    tune_inner_folds: int = 5
    cv_folds: int = 5
    cv_repeats: int = 10
    spatial_block_km: float | None = 10.0
    spatial_repeats: int = 5
    nested_cv: bool = True
    tune_cv: str = "random"             # inner CV of the tuner: random | spatial
    # reliability diagnostics (area of applicability, uncertainty, artefacts)
    reliability: bool = True
    reliability_methods: list | None = None   # None = every method
    reliability_folds: int = 5
    # outputs
    make_maps: bool = True
    classification: str = "jenks"
    n_classes: int = 5
    n_jobs: int = -1
    resume: bool = True                 # reuse finished steps in output_dir

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        known = set(RunConfig.__dataclass_fields__)
        return RunConfig(**{k: v for k, v in d.items() if k in known})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class Run:
    def __init__(self, cfg: RunConfig, log=print, progress=None):
        self.cfg = cfg
        self.log = log
        self.progress = progress or (lambda *_: None)
        self.out = cfg.output_dir
        for d in ("factors", "maps", "models", "tables", "figures", "_work"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.report = {"software": "LSM Mapper", "version": _version(),
                       "python": platform.python_version(),
                       "packages": _package_versions(),
                       "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "config": asdict(cfg), "notes": []}
        rp = os.path.join(self.out, "run_report.json")
        self.prev = {}
        if cfg.resume and os.path.exists(rp):
            with open(rp) as f:
                self.prev = json.load(f)
            if not _same_settings(self.prev.get("config", {}), asdict(cfg)):
                self.prev = {}          # settings changed: start again

    # ------------------------------------------------------------------ factors
    def _resume_factors(self):
        cfg = self.cfg
        if "grid" not in self.prev:
            return False
        fdir = os.path.join(self.out, "factors")
        names = [k for k in self.prev.get("factor_names", [])]
        if not names or not all(os.path.exists(os.path.join(fdir, f"{k}.tif")) for k in names):
            return False
        self.grid, notes = raster.reference_grid(cfg.dem, cfg.target_crs, cfg.target_res)
        self.factors = {k: os.path.join(fdir, f"{k}.tif") for k in names}
        self.categorical = list(self.prev.get("categorical", []))
        self.valid = raster.read_band(os.path.join(self.out, "_work", "valid_mask.tif")) > 0
        self.slope = raster.read_band(os.path.join(fdir, "Slope.tif"))
        for k in ("notes", "layer_checks", "grid", "factor_names", "categorical"):
            self.report[k] = self.prev[k]
        self.log("Resumed: factors already prepared.")
        return True

    def prepare_factors(self):
        cfg = self.cfg
        if self._resume_factors():
            return
        self.grid, notes = raster.reference_grid(cfg.dem, cfg.target_crs, cfg.target_res)
        self.report["notes"] += notes
        g = self.grid
        self.log(f"Grid: {g.crs.to_string()}  {g.res:.1f} m  {g.width} x {g.height} px")
        fdir = os.path.join(self.out, "factors")
        dem_al = os.path.join(fdir, "Elevation.tif")
        raster.align_to_grid(cfg.dem, g, dem_al, resampling=raster.Resampling.bilinear)
        dem = raster.read_band(dem_al)
        self.factors, self.categorical, checks = {}, [], []
        wanted = set(cfg.dem_factors)
        if "Elevation" in wanted:
            self.factors["Elevation"] = dem_al
        self.log("Deriving slope, aspect and curvatures (Horn; Zevenbergen-Thorne)...")
        surf = terrain.surface_derivatives(dem, g.res)
        for k, v in surf.items():
            if k in wanted:
                self.factors[k] = raster.write_array(v, g, os.path.join(fdir, f"{k}.tif"))
        if wanted & {"TWI", "SPI", "Dist_Drainage"}:
            self.log("Filling depressions and routing flow (Priority-Flood, D8)...")
            hyd = terrain.hydrology(dem, surf["Slope"], g.res, cfg.stream_area_km2)
            for k in ("TWI", "SPI", "Dist_Drainage"):
                if k in wanted:
                    self.factors[k] = raster.write_array(hyd[k], g, os.path.join(fdir, f"{k}.tif"))
            raster.write_array(hyd["_flow_acc"], g, os.path.join(self.out, "_work", "flow_acc.tif"))
            del hyd
        self.slope = surf["Slope"]
        del surf
        for spec in cfg.factors:
            spec = FactorSpec(**spec) if isinstance(spec, dict) else spec
            self.log(f"Aligning {spec.name}...")
            if spec.kind == "distance":
                feat = raster.rasterize_mask(spec.paths[0], g)
                arr = terrain.distance_to(feat, g.res, valid=np.isfinite(dem))
                self.factors[spec.name] = raster.write_array(arr, g, os.path.join(fdir, f"{spec.name}.tif"))
                checks.append({"name": spec.name, "kind": "distance", "path": spec.paths[0],
                               "features_cells": int(feat.sum())})
                continue
            cat = spec.kind == "categorical"
            tile_checks = [raster.inspect_layer(spec.name, p, g, cat) for p in spec.paths]
            chk = tile_checks[0]
            if len(tile_checks) > 1:   # tiles: coverage is the union of the tiles
                chk.overlap_pct = round(min(100.0, sum(t.overlap_pct for t in tile_checks)), 2)
                chk.errors = [e for e in chk.errors if "Covers only" not in e]
                chk.warnings = [w for w in chk.warnings if "Covers only" not in w]
                if chk.overlap_pct < 99:
                    chk.warnings.append(f"Tiles cover {chk.overlap_pct}% of the DEM extent.")
            checks.append(chk.as_dict())
            for e in chk.errors:
                raise ValueError(f"{spec.name}: {e}")
            for w in chk.warnings:
                self.log(f"  ! {spec.name}: {w}")
            rs = getattr(raster.Resampling, spec.resampling) if spec.resampling else None
            out_f = os.path.join(fdir, f"{spec.name}.tif")
            raster.align_tiles(spec.paths, g, out_f, categorical=cat, resampling=rs,
                               smooth_to_m=spec.smooth_to_m,
                               work=os.path.join(self.out, "_work"))
            self.factors[spec.name] = out_f
            if cat:
                self.categorical.append(spec.name)
        for x in cfg.exclude_factors:
            self.factors.pop(x, None)
        self.report["layer_checks"] = checks
        # valid-cell mask: inside ROI and every factor defined
        valid = np.isfinite(dem)
        if cfg.roi:
            valid &= raster.rasterize_mask(cfg.roi, g)
        for p in self.factors.values():
            valid &= np.isfinite(raster.read_band(p))
        self.valid = valid
        raster.write_array(valid.astype("uint8"), g, os.path.join(self.out, "_work", "valid_mask.tif"),
                           dtype="uint8", nodata=0)
        n = int(valid.sum())
        self.report["factor_names"] = list(self.factors)
        self.report["categorical"] = list(self.categorical)
        self.report["grid"] = {"crs": g.crs.to_string(), "res_m": g.res, "width": g.width,
                               "height": g.height, "valid_cells": n,
                               "valid_area_km2": n * g.res ** 2 / 1e6}
        self.log(f"Valid cells: {n:,} ({n * g.res ** 2 / 1e6:,.1f} km2)")

    # ------------------------------------------------------------------ samples
    def build_samples(self):
        cfg, g = self.cfg, self.grid
        ls_rc, info = sampling.load_inventory(cfg.inventory, g, self.valid)
        self.log(f"Inventory: {info}")
        n_abs = int(round(len(ls_rc) * cfg.absence_ratio))
        nls_rc, ainfo = sampling.sample_absence(
            g, self.valid, ls_rc, n_abs, seed=cfg.seed, slope=self.slope,
            min_slope=cfg.absence_min_slope, buffer_m=cfg.absence_buffer_m,
            region=cfg.absence_region)
        self.ls_rc = ls_rc
        df = sampling.build_table(self.factors, g, ls_rc, nls_rc)
        df = df.dropna()
        self.df = df
        self.names = list(self.factors)
        df.to_csv(os.path.join(self.out, "tables", "samples.csv"), index=False)
        slopes = self.slope[np.r_[ls_rc[:, 0], nls_rc[:, 0]], np.r_[ls_rc[:, 1], nls_rc[:, 1]]]
        self.report["samples"] = {"inventory": info, "absence": ainfo, "rows": int(len(df)),
                                  "landslide": int(df.label.sum()),
                                  "non_landslide": int((df.label == 0).sum()),
                                  "mean_slope_landslide": float(np.nanmean(slopes[:len(ls_rc)])),
                                  "mean_slope_non_landslide": float(np.nanmean(slopes[len(ls_rc):]))}
        # area sample for bivariate methods and multicollinearity
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.choice(np.flatnonzero(self.valid), size=min(500_000, int(self.valid.sum())), replace=False)
        rr, cc = np.unravel_index(idx, self.valid.shape)
        area = {}
        for k, p in self.factors.items():
            area[k] = raster.read_band(p)[rr, cc]
        self.area = pd.DataFrame(area)
        del self.slope

    def multicollinearity(self):
        s = self.area.sample(5000, random_state=self.cfg.seed)
        vif = validation.vif_table(s)
        corr = s.corr(method="pearson")
        vif.to_csv(os.path.join(self.out, "tables", "vif.csv"), index=False)
        corr.to_csv(os.path.join(self.out, "tables", "pearson.csv"))
        self.report["vif"] = dict(zip(vif.Factor, vif.VIF.round(3)))
        if self.cfg.vif_screen:
            keep, dropped = validation.vif_screen(s, self.cfg.vif_screen)
            self.report["vif_screen"] = {"threshold": self.cfg.vif_screen, "kept": keep, "dropped": dropped}
            self.log(f"VIF screening dropped: {[d[0] for d in dropped]}")
            self.names = keep

    # ------------------------------------------------------------------ models
    def model_all(self):
        cfg = self.cfg
        X = self.df[self.names].to_numpy(float)
        y = self.df.label.to_numpy(int)
        xy = self.df[["x", "y"]].to_numpy(float)
        tr, te = validation.holdout(X, y, cfg.test_size, cfg.seed)
        self.split = (tr, te)
        groups = (validation.spatial_blocks(xy, (cfg.spatial_block_km or 10.0) * 1000)
                  if cfg.tune_cv == "spatial" else None)
        self.report["split"] = {"train": int(len(tr)), "test": int(len(te))}
        self.report["methods"] = {}
        self.fitted = {}
        roc = {}
        cats = [c for c in self.categorical if c in self.names]
        prev_m = self.prev.get("methods", {}) if self.prev.get("split") == self.report["split"] else {}
        for mi, m in enumerate(cfg.methods):
            self.log(f"--- {m} ---")
            mf = os.path.join(self.out, "models", f"{_slug(m)}.joblib")
            if m in prev_m and "cv_kfold" in prev_m[m] and os.path.exists(mf):
                pipe = joblib.load(mf)["pipeline"]
                p = pipe.predict_proba(X[te])[:, 1]
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y[te], p)
                roc[m] = (fpr, tpr, prev_m[m]["test"]["AUC"])
                self.fitted[m] = pipe
                self.report["methods"][m] = prev_m[m]
                self.log("Resumed: already fitted and validated.")
                continue
            mk = lambda m=m: make_model(m, self.names, cats, cfg.params.get(m),
                                        area_sample=self.area, seed=cfg.seed, n_jobs=cfg.n_jobs)
            entry = {}
            pipe = mk()
            tune_kw = dict(search=cfg.tune_search, n_iter=cfg.tune_iter,
                           inner_folds=cfg.tune_inner_folds, n_jobs=cfg.n_jobs,
                           space=(cfg.search_spaces or {}).get(m) or None,
                           spatial=cfg.tune_cv == "spatial")
            if cfg.tune:
                self.log(f"Tuning on the training set ({cfg.tune_search} search, "
                         f"{cfg.tune_inner_folds}-fold {cfg.tune_cv} inner CV)...")
                search = validation.tuner(pipe, m, seed=cfg.seed, **tune_kw)
                search.fit(X[tr], y[tr], **({"groups": groups[tr]} if groups is not None else {}))
                pipe = search.best_estimator_
                entry["tuning"] = {"best_inner_auc": float(search.best_score_),
                                   "best_params": {k.replace("clf__", ""): _py(v)
                                                   for k, v in search.best_params_.items()}}
            else:
                pipe.fit(X[tr], y[tr])
            # default-parameter reference on the same split
            ref = mk().fit(X[tr], y[tr])
            p_ref = ref.predict_proba(X[te])[:, 1]
            entry["default_params"] = params_of(ref)
            entry["test_default"] = validation.metrics(y[te], p_ref, thresholded=m not in STAT_METHODS)
            if m in STAT_METHODS:
                pipe.named_steps["clf"].calibrate_output(self.area[self.names].to_numpy(float))
            p = pipe.predict_proba(X[te])[:, 1]
            entry["params"] = params_of(pipe)
            entry["test"] = validation.metrics(y[te], p, thresholded=m not in STAT_METHODS)
            p_tr = pipe.predict_proba(X[tr])[:, 1]
            entry["train_auc"] = float(validation.roc_auc_score(y[tr], p_tr))
            self.log(f"Test AUC {entry['test']['AUC']:.4f} (train {entry['train_auc']:.4f}; "
                     f"default params {entry['test_default']['AUC']:.4f})")
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y[te], p)
            roc[m] = (fpr, tpr, entry["test"]["AUC"])
            # cross-validation
            make_cv = (lambda: mk()) if not cfg.tune else (lambda: mk())
            for scheme in ("kfold", "spatial"):
                if scheme == "spatial" and not cfg.spatial_block_km:
                    continue
                self.log(f"{'Repeated' if scheme == 'kfold' else 'Spatial block'} CV...")
                cvdf = validation.cross_validate(
                    make_cv, m, X, y, scheme=scheme, k=cfg.cv_folds,
                    repeats=cfg.cv_repeats if scheme == "kfold" else cfg.spatial_repeats,
                    block_m=(cfg.spatial_block_km or 0) * 1000, xy=xy,
                    tune=cfg.tune and cfg.nested_cv, tune_kw=dict(tune_kw, n_iter=min(cfg.tune_iter, 20)),
                    seed=cfg.seed)
                cvdf.to_csv(os.path.join(self.out, "tables", f"cv_{scheme}_{_slug(m)}.csv"), index=False)
                entry[f"cv_{scheme}"] = validation.summarise(cvdf)
                a = entry[f"cv_{scheme}"]["AUC"]
                self.log(f"  AUC {a['mean']:.4f} +/- {a['sd']:.4f}")
            if m not in STAT_METHODS:
                imp = validation.importance(pipe, X[te], y[te], self.names, m, seed=cfg.seed)
                imp.to_csv(os.path.join(self.out, "tables", f"importance_{_slug(m)}.csv"), index=False)
                entry["importance"] = imp.round(5).to_dict(orient="list")
            else:
                tabs = pipe.named_steps["clf"].tables_
                pd.concat({k: v for k, v in tabs.items()}).to_csv(
                    os.path.join(self.out, "tables", f"class_weights_{_slug(m)}.csv"))
            joblib.dump({"pipeline": pipe, "features": self.names, "categorical": cats,
                         "method": m}, os.path.join(self.out, "models", f"{_slug(m)}.joblib"))
            self.fitted[m] = pipe
            self.report["methods"][m] = entry
            self._save_report()
            self.progress(mi + 1, len(cfg.methods))
        self.roc = roc

    # ------------------------------------------------------------------ maps
    def predict_maps(self, block_rows=256):
        g, cfg = self.grid, self.cfg
        srcs = {k: rasterio.open(self.factors[k]) for k in self.names}
        try:
            for m, pipe in self.fitted.items():
                out_p = os.path.join(self.out, "maps", f"LSM_{_slug(m)}_probability.tif")
                if "map" in self.report["methods"][m] and os.path.exists(out_p):
                    self.log(f"Resumed: {m} map exists.")
                    continue
                self.log(f"Predicting {m} map...")
                prob = np.full(g.shape, np.nan, np.float32)
                for r0 in range(0, g.height, block_rows):
                    h = min(block_rows, g.height - r0)
                    w = Window(0, r0, g.width, h)
                    vm = self.valid[r0:r0 + h]
                    if not vm.any():
                        continue
                    cols = [raster.clean_nodata(srcs[k].read(1, window=w).astype(np.float32),
                                                srcs[k].nodata)[vm] for k in self.names]
                    Xb = np.stack(cols, 1)
                    blk = prob[r0:r0 + h]
                    blk[vm] = pipe.predict_proba(Xb)[:, 1]
                raster.write_array(prob, g, out_p)
                ub = classify.breaks(prob, cfg.classification, cfg.n_classes, seed=cfg.seed)
                cls = classify.apply(prob, ub)
                raster.write_array(np.nan_to_num(cls, nan=0).astype("uint8"), g,
                                   os.path.join(self.out, "maps", f"LSM_{_slug(m)}_classes.tif"),
                                   dtype="uint8", nodata=0)
                e = self.report["methods"][m]
                if "Slope" in self.factors:
                    e["flat_terrain_check"] = classify.flat_terrain_check(cls, raster.read_band(self.factors["Slope"]))
                e["map"] = {"classification": cfg.classification, "upper_breaks": ub.tolist(),
                            "classes": classify.class_table(cls, g.res ** 2, cfg.n_classes),
                            "landslide_share_pct": classify.landslide_density(cls, self.ls_rc, cfg.n_classes)}
                for meth in ("equal", "quantile", "jenks"):
                    if meth == cfg.classification:
                        continue
                    b2 = classify.breaks(prob, meth, cfg.n_classes, seed=cfg.seed)
                    e["map"][f"classes_{meth}"] = classify.class_table(classify.apply(prob, b2), g.res ** 2,
                                                                       cfg.n_classes)
                self._save_report()
        finally:
            for s in srcs.values():
                s.close()

    # ------------------------------------------------------------------ reliability
    def _block_iter(self, block_rows=256):
        g = self.grid
        srcs = {k: rasterio.open(self.factors[k]) for k in self.names}
        try:
            for r0 in range(0, g.height, block_rows):
                h = min(block_rows, g.height - r0)
                vm = self.valid[r0:r0 + h]
                if not vm.any():
                    continue
                w = Window(0, r0, g.width, h)
                cols = [raster.clean_nodata(srcs[k].read(1, window=w).astype(np.float32),
                                            srcs[k].nodata)[vm] for k in self.names]
                yield r0, h, vm, np.stack(cols, 1)
        finally:
            for s in srcs.values():
                s.close()

    def _edge_layers(self):
        """Raw (not de-blocked) versions of coarse input layers, aligned with
        nearest-neighbour resampling, used to locate their block edges."""
        cfg, g = self.cfg, self.grid
        names = reliability.coarse_layers(self.report.get("layer_checks"), g.res)
        out = {}
        for spec in cfg.factors:
            spec = FactorSpec(**spec) if isinstance(spec, dict) else spec
            if spec.name not in names or spec.name not in self.names:
                continue
            p = os.path.join(self.out, "_work", f"edges_{spec.name}.tif")
            if not os.path.exists(p):
                raster.align_tiles(spec.paths, g, p, categorical=True,
                                   resampling=raster.Resampling.nearest, smooth_to_m=None,
                                   work=os.path.join(self.out, "_work"))
            out[spec.name] = p
        return out

    def reliability(self):
        """Area of applicability, accuracy against dissimilarity, CV-ensemble
        uncertainty and block-edge artefact index for each method's map."""
        cfg, g = self.cfg, self.grid
        methods = [m for m in (cfg.reliability_methods or list(self.fitted)) if m in self.fitted]
        X = self.df[self.names].to_numpy(float)
        y = self.df.label.to_numpy(int)
        xy = self.df[["x", "y"]].to_numpy(float)
        tr, _ = self.split
        block = (cfg.spatial_block_km or 10.0) * 1000
        k = cfg.reliability_folds
        cats = [c for c in self.categorical if c in self.names]
        fold_all = reliability.spatial_folds(xy, y, block, k, cfg.seed)
        fold_tr = reliability.spatial_folds(xy[tr], y[tr], block, k, cfg.seed)
        area_X = self.area[self.names].to_numpy(float)
        prev = self.prev.get("reliability", {}) if self.prev.get("split") == self.report.get("split") else {}
        rep = self.report.setdefault("reliability", {})
        rep["settings"] = {"folds": k, "block_km": block / 1000,
                           "weights": "permutation importance (equal weights if unavailable)",
                           "threshold": "upper whisker of cross-validated training DI"}
        # weights for methods without an importance table: best ML method
        ml = {m: e for m, e in self.report["methods"].items() if e.get("importance")}
        best_ml = max(ml, key=lambda m: ml[m].get("cv_spatial", ml[m]["cv_kfold"])["AUC"]["mean"]) if ml else None
        edges = self._edge_layers()
        edge_arr = {n: raster.read_band(p) for n, p in edges.items()}
        for m in methods:
            files = {kk: os.path.join(self.out, "maps", f"{kk}_{_slug(m)}.tif")
                     for kk in ("AOA_DI", "AOA_mask", "Uncertainty_SD")}
            if m in prev and all(os.path.exists(f) for f in files.values()):
                rep[m] = prev[m]
                self.log(f"Resumed: {m} reliability exists.")
                continue
            self.log(f"Reliability: {m}...")
            w_src = m
            w = reliability.importance_weights(self.report["methods"][m], self.names)
            if w is None and best_ml:
                w_src, w = best_ml, reliability.importance_weights(self.report["methods"][best_ml], self.names)
            space = reliability.FeatureSpace(self.names, cats, w)
            aoa = reliability.AOA(space).fit(X[tr], fold_tr)
            pipe = self.fitted[m]
            models, oof = reliability.fold_models(pipe, m, X, y, fold_all, area_X)
            di_oof = reliability.fold_di(self.names, cats, w, X, fold_all)
            by_di = reliability.auc_by_di(y, oof, di_oof, aoa.threshold_)
            pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "label": y, "fold": fold_all,
                          "oof_probability": oof, "DI": di_oof}).to_csv(
                os.path.join(self.out, "tables", f"oof_spatial_{_slug(m)}.csv"), index=False)
            pd.DataFrame(by_di).to_csv(os.path.join(self.out, "tables", f"auc_by_di_{_slug(m)}.csv"),
                                       index=False)
            DI = np.full(g.shape, np.nan, np.float32)
            SD = np.full(g.shape, np.nan, np.float32)
            for r0, h, vm, Xb in self._block_iter():
                DI[r0:r0 + h][vm] = aoa.di(Xb)
                P = np.stack([mm.predict_proba(Xb)[:, 1] for mm in models], 0)
                SD[r0:r0 + h][vm] = P.std(0)
            mask = np.where(np.isfinite(DI), np.where(DI <= aoa.threshold_, 1, 2), 0).astype("uint8")
            raster.write_array(DI, g, files["AOA_DI"])
            raster.write_array(mask, g, files["AOA_mask"], dtype="uint8", nodata=0)
            raster.write_array(SD, g, files["Uncertainty_SD"])
            cls = raster.read_band(os.path.join(self.out, "maps", f"LSM_{_slug(m)}_classes.tif"))
            cls[cls == 0] = np.nan
            prob = raster.read_band(os.path.join(self.out, "maps", f"LSM_{_slug(m)}_probability.tif"))
            ls_out = mask[self.ls_rc[:, 0], self.ls_rc[:, 1]]
            hv = np.isin(cls, (4, 5))
            entry = {"weights_from": w_src,
                     "weights": dict(zip(self.names, np.round(space.weights, 4).tolist())),
                     "threshold": aoa.threshold_, "dbar": aoa.dbar_,
                     "outside_aoa_pct": float(100 * (mask == 2).sum() / max((mask > 0).sum(), 1)),
                     "landslides_outside_pct": float(100 * (ls_out == 2).mean()),
                     "high_vhigh_outside_pct": float(100 * (hv & (mask == 2)).sum() / max(hv.sum(), 1)),
                     "outside_by_class_pct": [float(100 * ((cls == c) & (mask == 2)).sum() /
                                                    max((cls == c).sum(), 1)) for c in range(1, cfg.n_classes + 1)],
                     "oof_spatial_auc": float(validation.roc_auc_score(y, oof)),
                     "auc_by_di": by_di,
                     "uncertainty_sd_mean": float(np.nanmean(SD)),
                     "uncertainty_sd_p90": float(np.nanquantile(SD, 0.9)),
                     "uncertainty_by_class": reliability.class_summary(cls, SD, cfg.n_classes),
                     "di_by_class": reliability.class_summary(cls, DI, cfg.n_classes),
                     "artefact_index": {}}
            for n, la in edge_arr.items():
                idx, npairs = reliability.artefact_index(prob, la)
                entry["artefact_index"][n] = {"index": idx, "edge_pairs": npairs}
            if "Slope" in self.factors:
                entry["flat_terrain_check"] = self.report["methods"][m].get("flat_terrain_check")
            rep[m] = entry
            del DI, SD, mask, cls, prob
            self.log(f"  outside AOA {entry['outside_aoa_pct']:.1f}% of area; "
                     f"mean SD {entry['uncertainty_sd_mean']:.3f}; "
                     f"artefact {({n: round(v['index'], 3) for n, v in entry['artefact_index'].items() if v['index']})}")
            self._save_report()

    # ------------------------------------------------------------------ run
    def run(self):
        t0 = time.time()
        steps = [("Preparing factors", self.prepare_factors), ("Sampling", self.build_samples),
                 ("Multicollinearity", self.multicollinearity), ("Modelling", self.model_all)]
        if self.cfg.make_maps:
            steps.append(("Mapping", self.predict_maps))
            if self.cfg.reliability:
                steps.append(("Reliability", self.reliability))
        for label, fn in steps:
            self.log(f"== {label} ==")
            fn()
            self._save_report()
        from . import figures
        figures.all_figures(self)
        self.report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.report["runtime_s"] = round(time.time() - t0, 1)
        self._save_report()
        shutil.rmtree(os.path.join(self.out, "_work", "tmp"), ignore_errors=True)
        self.log(f"Done in {self.report['runtime_s'] / 60:.1f} min. Report: run_report.json")
        return self.report

    def _save_report(self):
        with open(os.path.join(self.out, "run_report.json"), "w") as f:
            json.dump(self.report, f, indent=2, default=_py)


# settings that only affect the reliability step, which re-runs on its own
_LATE_KEYS = {"reliability", "reliability_methods", "reliability_folds", "resume", "n_jobs"}


def _same_settings(old: dict, new: dict) -> bool:
    """True if a previous run used the same settings. Keys added in later
    versions count as unchanged when the new value is their default."""
    defaults = {}
    for k, f in RunConfig.__dataclass_fields__.items():
        if f.default is not dataclasses.MISSING:
            defaults[k] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[k] = f.default_factory()
    for k, v in new.items():
        if k in _LATE_KEYS:
            continue
        if k not in old:
            if v != defaults.get(k):
                return False
        elif old[k] != v:
            return False
    return True


def _slug(m):
    return m.replace(" ", "_")


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v) if not isinstance(v, (int, float, str, bool, type(None), list, dict)) else v


def _package_versions():
    import importlib.metadata as md
    out = {}
    for p in ("numpy", "pandas", "scipy", "rasterio", "geopandas", "scikit-learn", "xgboost",
              "numba", "jenkspy"):
        try:
            out[p] = md.version(p)
        except Exception:
            pass
    return out


def _version():
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "dev"
