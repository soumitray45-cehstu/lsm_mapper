import os

def authenticate_gee():
    import ee
    try:
        ee.Initialize()
        print("GEE initialized successfully.")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        print("GEE authenticated and initialized.")

def fetch_ndvi(bbox, start_date, end_date, out_dir):
    import ee
    os.makedirs(out_dir, exist_ok=True)
    region = ee.Geometry.Rectangle(bbox)
    img = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .median()
        .normalizedDifference(["B8", "B4"])
        .rename("NDVI")
    )
    task = ee.batch.Export.image.toDrive(
        image=img,
        description="NDVI",
        folder="LSM_outputs",
        region=region,
        scale=30,
        fileFormat="GeoTIFF"
    )
    task.start()
    print("NDVI export started — check Google Drive > LSM_outputs folder.")

def fetch_lulc(bbox, out_dir):
    import ee
    os.makedirs(out_dir, exist_ok=True)
    region = ee.Geometry.Rectangle(bbox)
    img = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .clip(region)
    )
    task = ee.batch.Export.image.toDrive(
        image=img,
        description="LULC",
        folder="LSM_outputs",
        region=region,
        scale=30,
        fileFormat="GeoTIFF"
    )
    task.start()
    print("LULC export started — check Google Drive > LSM_outputs folder.")