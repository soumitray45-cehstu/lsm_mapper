# Case study: Chattogram Division, Bangladesh

Settings and scripts behind the results in the paper.

| File | Purpose |
|---|---|
| `chattogram_run.json` | Settings of the main run (load in the GUI or `--run`) |
| `results/run_report.json` | Every check, parameter and score of the main run |
| `results/samples.csv` | Factor values at the 1,390 landslide / non-landslide cells |
| `results/run_report_slope3_sampling.json` | Comparison run with non-landslide cells restricted to slopes > 3° |
| `sensitivity.py` | Factor-set, block-size and tuning sensitivity; variant maps |
| `artefacts.py`, `post.py` | Block-edge artefact index and flat-terrain check |

Input data (paths are relative to this folder, under `data/`):
* `DEM.tif`: SRTM 30 m (USGS/SRTMGL1_003), exported from Google Earth Engine, UTM 46N
* `Landslide_Inventory_UTM46N.shp`: Rabby & Li (2019), https://doi.org/10.3390/data5010004
* `NDVI-*.tif`: Sentinel-2 L2A median NDVI 2020–2022, 10 m tiles. The 30 m strips in
  `NDVI_tiles_30m/` were produced with LSM Mapper's own `raster.align_tiles` (mean
  resampling) to keep the transfer size small; giving the original 10 m tiles gives the same layer.
* `LULC-*.tif`: ESA WorldCover v200 (2021), 10 m tiles
* `Rainfall.tif`: CHIRPS v2.0 mean annual precipitation 2015–2022, exported at 30 m
* `Soil.tif`: OpenLandMap USDA texture class, 250 m

The input rasters (~1 GB) are archived with the Zenodo release of the software.
