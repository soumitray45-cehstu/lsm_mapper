"""Spatial versus random inner CV for hyper-parameter tuning (case study).

Re-uses the factors of the main run and refits XGBoost and Random Forest with
the tuner's inner folds formed by 10 km spatial blocks (tune_cv="spatial");
everything else is identical to the main run. Compare the nested spatial-CV
AUC with the main run (random inner folds).

    python case_study/tuning_cv.py MAIN_CONFIG.json OUT_DIR
"""
import json
import os
import sys

from lsm_mapper.core.pipeline import Run, RunConfig

main_cfg, out = sys.argv[1], sys.argv[2]
cfg = RunConfig.load(main_cfg)
main_out = cfg.output_dir
with open(os.path.join(main_out, "run_report.json")) as f:
    main_rep = json.load(f)
cfg.output_dir = out
cfg.tune_cv = "spatial"
cfg.methods = ["XGBoost", "Random Forest"]
cfg.cv_repeats = 2
cfg.make_maps = False
cfg.reliability = False
os.makedirs(out, exist_ok=True)
for d in ("factors",):
    if not os.path.exists(os.path.join(out, d)):
        os.symlink(os.path.join(main_out, d), os.path.join(out, d))
os.makedirs(os.path.join(out, "_work"), exist_ok=True)
vm = os.path.join(out, "_work", "valid_mask.tif")
if not os.path.exists(vm):
    os.symlink(os.path.join(main_out, "_work", "valid_mask.tif"), vm)

run = Run(cfg)
keep = {k: main_rep[k] for k in ("grid", "factor_names", "categorical", "notes", "layer_checks")}
if run.prev.get("methods") and run.prev.get("grid"):
    keep["methods"] = run.prev["methods"]
    keep["split"] = run.prev.get("split")
run.prev = keep
run.prepare_factors()
run.build_samples()
run.multicollinearity()
run.model_all()
run._save_report()
for m, e in run.report["methods"].items():
    print(m, "spatial-tuned: nested spatial CV AUC", e["cv_spatial"]["AUC"], "test", e["test"]["AUC"])
