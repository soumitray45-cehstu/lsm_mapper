# LSM Mapper 🗺️

**A free, standalone desktop application for automated 
Landslide Susceptibility Mapping (LSM)**

No GIS software required. No coding knowledge needed.
Upload a DEM + landslide inventory → get a full 
susceptibility map in minutes.

---

## Features

- ✅ **6 Methods**: Random Forest, XGBoost, 
  Logistic Regression, Frequency Ratio, 
  Weights of Evidence, Information Value
- ✅ **Auto-derives 8 terrain factors** from DEM
- ✅ **AUC-ROC validation** with ROC curve chart
- ✅ **Feature importance** analysis
- ✅ **Interactive HTML map** output
- ✅ **5-class susceptibility map** (Very Low → Very High)
- ✅ **Completely free** — no license required
- ✅ **Standalone .exe** — no installation needed

---

## Download

👉 [Download LSM Mapper v1.0 (.exe)](releases)

---

## How to Use

1. Download and run `LSM_Mapper.exe`
2. Upload your **DEM** (GeoTIFF)
3. Upload your **Landslide Inventory** (Shapefile)
4. Select a method (e.g. XGBoost)
5. Click **Train Model**
6. View your susceptibility map

---

## Supported Input Formats

| Data | Format |
|---|---|
| DEM | GeoTIFF (.tif) |
| Landslide Inventory | Shapefile (.shp) |
| Optional factors | GeoTIFF (.tif) |

---

## Optional Conditioning Factors

Upload any of these to improve accuracy:
- Soil type, Geology, LULC, Rainfall, NDVI
- Distance to Roads, Drainage, Faults

---

## Validation — Chattogram Division, Bangladesh

| Method | AUC-ROC |
|---|---|
| XGBoost | **0.9251** ⭐ |
| Random Forest | 0.8501 |
| Weights of Evidence | 0.8322 |
| Frequency Ratio | 0.8240 |
| Information Value | 0.8164 |
| Logistic Regression | 0.7580 |

---

## System Requirements

- Windows 10/11 (64-bit)
- 8 GB RAM minimum
- 2 GB free disk space

---

## Citation

If you use LSM Mapper in your research, please cite:

> [Your Name] et al. (2025). LSM Mapper: A GIS-Independent 
> Desktop Application for Automated Landslide Susceptibility 
> Mapping. *Natural Hazards and Earth System Sciences*.
> DOI: [to be added]

---

## Built With

Python 3.11 · PyQt5 · scikit-learn · XGBoost · 
rasterio · geopandas · folium · matplotlib

---

## License

MIT License — free for academic and commercial use.
