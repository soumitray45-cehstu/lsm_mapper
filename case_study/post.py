"""Post-run checks for the case study: flat-terrain check for both sampling
designs and block-edge artefact indices for the main and variant maps.

    python case_study/post.py MAIN ALT SENS
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsm_mapper.core import raster, classify  # noqa: E402

main, alt, sens = sys.argv[1:4]
slope = raster.read_band(os.path.join(main, "factors", "Slope.tif"))
out = {}
for tag, d in (("whole_area", main), ("slope_gt3_bbox", alt)):
    for m in ("XGBoost", "Random_Forest", "Logistic_Regression", "Frequency_Ratio",
              "Weights_of_Evidence", "Information_Value"):
        p = os.path.join(d, "maps", f"LSM_{m}_classes.tif")
        if os.path.exists(p):
            out[f"{tag}|{m.replace('_', ' ')}"] = classify.flat_terrain_check(raster.read_band(p), slope)
json.dump(out, open(os.path.join(sens, "flat_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

maps = []
for m in ("XGBoost", "Random_Forest", "Logistic_Regression", "Frequency_Ratio"):
    maps.append(f"{m}|main={main}/maps/LSM_{m}_probability.tif")
for m in ("XGBoost", "Random_Forest"):
    for v in ("raw_rain", "no_rain_soil"):
        maps.append(f"{m}|{v}={sens}/LSM_{m}_{v}_probability.tif")
cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "artefacts.py"), os.path.join(sens, "artefacts.csv"),
       f"rain={sens}/Rainfall_raw_aligned.tif", f"soil={main}/factors/Soil.tif", "--"] + maps
subprocess.run(cmd, check=True)
