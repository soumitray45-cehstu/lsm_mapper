"""Restore the main run after another configuration wrote into its folder:
restore the report and sample table, recompute the XGBoost k-fold CV table,
and rebuild the reliability entries from the saved rasters."""
import json
import shutil
import sys

import numpy as np
import pandas as pd

from lsm_mapper.core import validation
from lsm_mapper.core.models import make_model
from lsm_mapper.core.pipeline import Run, RunConfig

C = "/home/lsm/case"
shutil.copy(f"{C}/main_run_report_backup.json", f"{C}/main/run_report.json")
shutil.copy("/home/lsm/repo/lsm_mapper/case_study/results/samples.csv", f"{C}/main/tables/samples.csv")
cfg = RunConfig.load(f"{C}/main.json")
cfg.reliability_reuse_maps = ["XGBoost", "Random Forest", "Logistic Regression", "Frequency Ratio",
                              "Weights of Evidence", "Information Value"]
run = Run(cfg)
assert run.prev.get("methods"), "resume failed"
run.prepare_factors()
run.build_samples()
run.multicollinearity()
run.model_all()
# XGBoost k-fold table (overwritten); same call as model_all, deterministic
m = "XGBoost"
X = run.df[run.names].to_numpy(float)
y = run.df.label.to_numpy(int)
xy = run.df[["x", "y"]].to_numpy(float)
cats = [c for c in run.categorical if c in run.names]
mk = lambda: make_model(m, run.names, cats, cfg.params.get(m), area_sample=run.area, seed=cfg.seed, n_jobs=cfg.n_jobs)
tune_kw = dict(search=cfg.tune_search, n_iter=min(cfg.tune_iter, 20), inner_folds=cfg.tune_inner_folds,
               n_jobs=cfg.n_jobs, space=(cfg.search_spaces or {}).get(m) or None, spatial=False)
cvdf = validation.cross_validate(mk, m, X, y, scheme="kfold", k=cfg.cv_folds, repeats=cfg.cv_repeats,
                                 block_m=(cfg.spatial_block_km or 0) * 1000, xy=xy, tune=True,
                                 tune_kw=tune_kw, seed=cfg.seed)
want = run.report["methods"][m]["cv_kfold"]["AUC"]["mean"]
print("kfold mean", cvdf.AUC.mean(), "report", want, flush=True)
cvdf.to_csv(f"{C}/main/tables/cv_kfold_XGBoost.csv", index=False)
run.predict_maps()
run.reliability()
from lsm_mapper.core import figures
figures.all_figures(run)
run._save_report()
print("repaired")
