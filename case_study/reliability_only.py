"""Run only the reliability step on a finished run made with an earlier
version of LSM Mapper (whose report lacks the keys used for resuming).

    python case_study/reliability_only.py CONFIG.json
"""
import json
import os
import sys

from lsm_mapper.core.pipeline import Run, RunConfig

cfg = RunConfig.load(sys.argv[1])
run = Run(cfg)
with open(os.path.join(cfg.output_dir, "run_report.json")) as f:
    old = json.load(f)
fdir = os.path.join(cfg.output_dir, "factors")
names = ["Elevation", "Slope", "Aspect", "Curvature", "Plan_Curvature", "Profile_Curvature", "TWI", "SPI",
         "Dist_Drainage"] + [s["name"] for s in cfg.factors]
names = [n for n in names if os.path.exists(os.path.join(fdir, f"{n}.tif"))]
old.setdefault("factor_names", names)
old.setdefault("categorical", [s["name"] for s in cfg.factors if s.get("kind") == "categorical"])
run.prev = old
run.prepare_factors()
run.build_samples()
run.multicollinearity()
run.model_all()
run.report["methods"] = {m: old["methods"][m] for m in run.report["methods"]}
run.reliability()
for k in ("grid", "samples", "vif", "split", "methods", "finished", "runtime_s"):
    if k in old:
        run.report[k] = old[k] if k != "methods" else run.report[k]
run._save_report()
print("done")
