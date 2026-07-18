import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def align_raster(src_path, dst_path, target_crs="EPSG:4326", target_res=0.0003):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds, resolution=target_res
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32"
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
    print(f"Aligned: {dst_path}")

def align_all(paths_dict, output_dir, target_crs="EPSG:4326"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    aligned = {}
    for name, path in paths_dict.items():
        out_path = os.path.join(output_dir, f"{name}_aligned.tif")
        align_raster(path, out_path, target_crs)
        aligned[name] = out_path
    return aligned