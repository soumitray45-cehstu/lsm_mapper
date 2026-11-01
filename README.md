# LSM Mapper

Free, open-source desktop software for **landslide susceptibility mapping (LSM)**.
It takes a DEM, a landslide inventory and any number of additional conditioning
factors, and produces validated susceptibility maps. You don't need a GIS or any
programming to use it.

* **Terrain factors from the DEM:** elevation, slope and aspect (Horn), curvature, plan
  and profile curvature (Zevenbergen & Thorne), TWI and SPI from depression-filled D8
  flow accumulation, and distance to DEM-derived drainage.
* **Any other factors**, as many as you need: continuous or categorical rasters,
  multi-tile exports (e.g. Google Earth Engine), and vector layers turned into
  distance surfaces.
* **Automatic alignment checks** for coordinate system, pixel size, extent and NoData
  (including float32 fill values). The resampling method follows the data type.
  Layers that are coarser than they look (up-sampled) are flagged, and can optionally
  be de-blocked.
* **Six methods:** Random Forest, XGBoost, Logistic Regression, Frequency Ratio,
  Weights of Evidence and Information Value.
* **Editable hyper-parameters**, plus random or grid **tuning** with inner
  cross-validation.
* **Validation** by hold-out, repeated stratified *k*-fold CV and **spatial block CV**; tuning with random or spatial inner folds.
* **Reliability diagnostics for every map**: area of applicability (dissimilarity index and mask, Meyer & Pebesma 2021), uncertainty map from the spatial-CV ensemble, AUC inside/outside the area of applicability and by dissimilarity, block-edge artefact index for coarse input layers, and a flat-terrain sampling check (`maps/AOA_DI_*.tif`, `maps/AOA_mask_*.tif`, `maps/Uncertainty_SD_*.tif`, `tables/auc_by_di_*.csv`, `figures/reliability.png`, and the `reliability` section of `run_report.json`).
  Nested tuning is available, so CV scores never see the tuning data.
* **Importance:** impurity (RF), gain (XGBoost) and permutation importance on the test set.
* **Multicollinearity:** VIF and Pearson correlation, with optional VIF-based factor screening.
* **Map classes** by Jenks natural breaks (default), equal interval or quantile, with
  class areas and the share of landslides in each class.
* **A run report** (`run_report.json`) records every input, check, parameter, score and
  package version, so any run can be repeated from its saved settings.

## Download (Windows)

Download `LSM_Mapper_v2.x_Windows_x64.zip` from the
[Releases](https://github.com/soumitray45-cehstu/lsm_mapper/releases) page, unzip it
and run `LSM_Mapper.exe`. There is nothing to install.
`LSM_Mapper_cli.exe --selftest` runs a short synthetic case to check the installation.

## Run from source

```bash
pip install -r requirements.txt
python main.py                      # desktop interface
python main.py --run settings.json  # batch run with saved settings
python -m pytest tests              # tests
```

## Workflow

1. **Data**: choose the DEM and the landslide inventory (points or polygons), and
   optionally a study-area boundary. Add further factors with *Add factor*.
2. **Settings**: choose the methods, then edit hyper-parameters and search spaces.
   Set the tuning, validation, non-landslide sampling and map classification options.
3. **Run**: the log shows every check and step.
4. **Results**: metrics table, maps, ROC curves, CV distributions, importance and
   correlation figures, plus an interactive web map.

Outputs (in the chosen folder):
`factors/` aligned factor rasters · `maps/` probability and class GeoTIFFs, plus
`interactive_map.html` · `tables/` samples, CV folds, importance, VIF and class weights ·
`figures/` 300 dpi PNG · `models/` fitted models · `run_report.json` · `run_config.json`.

## Case study

The settings and results for the Chattogram Division (Bangladesh) case study in the
accompanying paper are in [`case_study/`](case_study).

## Citation

Ray, S., Rahman, M. M., & Nuhash, M. M. R. LSM Mapper: an open-source desktop
application for landslide susceptibility mapping with machine learning and bivariate
statistical methods. *Mersin Photogrammetry Journal* (under review).
Software archive (v2.1.1): https://doi.org/10.5281/zenodo.23853999 (v1.0.1: https://doi.org/10.5281/zenodo.21427097)

## License

MIT
