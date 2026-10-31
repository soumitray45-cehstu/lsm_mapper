# Sample dataset

A 42 km x 42 km window of the Chattogram case study (WGS 84 / UTM 46N) for
trying LSM Mapper in a few minutes.

| File | Content |
|---|---|
| `DEM.tif` | SRTM 30 m DEM |
| `Landslide_inventory.shp` | 162 landslide points (Rabby & Li, 2019) |
| `NDVI_Sentinel2.tif` | Sentinel-2 NDVI, median 2020-2022, 30 m |
| `LULC_WorldCover.tif` | ESA WorldCover v200 (2021), 10 m, categorical |
| `Rainfall_CHIRPS.tif` | CHIRPS v2.0 mean annual rainfall 2015-2022, stored at 30 m (~5.5 km native; enable de-blocking) |
| `Soil_OpenLandMap.tif` | OpenLandMap USDA soil texture class, 250 m, categorical |

`sample_run.json` holds settings for a quick run. Load them with *File > Load settings*,
or run `LSM_Mapper_cli.exe --run sample_data/sample_run.json` from the program folder
(paths are relative to the current folder).
