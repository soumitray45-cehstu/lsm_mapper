import rasterio
import numpy as np

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, meta

def get_raster_info(path):
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "resolution": src.res,
            "shape": src.shape,
            "bounds": src.bounds
        }