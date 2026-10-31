"""Raster input, inspection and alignment.

Every conditioning factor is brought onto one reference grid (the DEM grid,
optionally re-projected to a metric CRS) before anything else happens.
The checks here answer one question per input: can this layer be stacked
with the DEM without silently corrupting values?
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds

# Values this large are fill values (e.g. float32 max written by ArcGIS/GEE),
# never real measurements of any conditioning factor.
FILL_ABS_LIMIT = 1e30

GTIFF_OPTS = dict(driver="GTiff", compress="deflate", predictor=3,
                  tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")


@dataclass
class Grid:
    crs: CRS
    transform: rasterio.Affine
    width: int
    height: int

    @property
    def res(self) -> float:
        return abs(self.transform.a)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def bounds(self):
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)

    def profile(self, dtype="float32", nodata=np.nan):
        p = dict(GTIFF_OPTS, crs=self.crs, transform=self.transform,
                 width=self.width, height=self.height, count=1,
                 dtype=dtype, nodata=nodata)
        if np.dtype(dtype).kind != "f":
            p["predictor"] = 2
        return p


@dataclass
class LayerCheck:
    name: str
    path: str
    crs: str | None
    epsg: int | None
    res: tuple
    shape: tuple
    dtype: str
    nodata: float | None
    categorical: bool
    overlap_pct: float = 100.0
    res_ratio: float = 1.0
    resampling: str = ""
    fill_values_found: int = 0
    effective_res_m: float | None = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def read_band(path: str, window=None) -> np.ndarray:
    """Read band 1 as float32 with every flavour of NoData set to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1, window=window).astype(np.float32)
        nd = src.nodata
    return clean_nodata(arr, nd)


def clean_nodata(arr: np.ndarray, nodata=None) -> np.ndarray:
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        arr[arr == np.float32(nodata)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) > FILL_ABS_LIMIT] = np.nan
    return arr


def mosaic_tiles(paths: list[str], out_path: str) -> str:
    """Merge tiled exports (e.g. Google Earth Engine 'name-0000000000-…' tiles)."""
    if len(paths) == 1:
        return paths[0]
    srcs = [rasterio.open(p) for p in paths]
    try:
        crs = {s.crs.to_string() for s in srcs}
        if len(crs) > 1:
            raise ValueError(f"Tiles have different CRS: {crs}")
        arr, transform = merge(srcs)
        prof = srcs[0].profile.copy()
        prof.update(GTIFF_OPTS, height=arr.shape[1], width=arr.shape[2],
                    transform=transform, count=1)
        if np.dtype(prof["dtype"]).kind != "f":
            prof["predictor"] = 2
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(arr[0], 1)
    finally:
        for s in srcs:
            s.close()
    return out_path


def reference_grid(dem_path: str, target_crs: str | None = None,
                   target_res: float | None = None) -> tuple[Grid, list[str]]:
    """Grid on which all factors are computed.

    A DEM in geographic coordinates is projected to its local UTM zone so that
    slope, curvature, flow length and distances are computed in metres.
    """
    notes = []
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError("The DEM has no coordinate reference system. "
                             "Assign one before using it.")
        src_crs, src_t, w, h = src.crs, src.transform, src.width, src.height
        bounds = src.bounds
    if target_crs:
        dst_crs = CRS.from_user_input(target_crs)
    elif src_crs.is_geographic:
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.bottom + bounds.top) / 2
        dst_crs = utm_crs_for(cx, cy)
        notes.append(f"DEM is in geographic coordinates ({src_crs.to_string()}); "
                     f"it will be projected to {dst_crs.to_string()} so that terrain "
                     f"derivatives are in metres.")
    else:
        dst_crs = src_crs
    if dst_crs.is_geographic:
        raise ValueError("Target CRS must be projected (metres).")
    if dst_crs == src_crs and target_res is None:
        return Grid(src_crs, src_t, w, h), notes
    if target_res is None:
        # keep approximately the native ground resolution
        t0, w0, h0 = calculate_default_transform(src_crs, dst_crs, w, h, *bounds)
        target_res = round(abs(t0.a))
    t, ww, hh = calculate_default_transform(src_crs, dst_crs, w, h, *bounds,
                                            resolution=target_res)
    notes.append(f"Reference grid: {dst_crs.to_string()}, {target_res} m, {ww} x {hh} px.")
    return Grid(dst_crs, t, ww, hh), notes


def effective_resolution(path: str, grid_res: float, max_px=2_000_000) -> float | None:
    """Estimate the true information resolution of a raster that may have been
    up-sampled before it reached us (e.g. 5.5 km CHIRPS exported at 30 m).

    Returns the median run length of identical values times the pixel size,
    or None if the layer looks genuinely continuous at its pixel size.
    """
    with rasterio.open(path) as src:
        step = max(1, int(math.sqrt(src.width * src.height / max_px)))
        out_shape = (max(1, src.height // step), max(1, src.width // step))
        arr = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float64)
        px = abs(src.transform.a) * step if not src.crs.is_geographic else None
    if px is None:
        return None
    arr = clean_nodata(arr.astype(np.float32), None)
    row = arr[arr.shape[0] // 2]
    runs = []
    for r in arr[:: max(1, arr.shape[0] // 50)]:
        r = r[np.isfinite(r)]
        if r.size < 10:
            continue
        change = np.flatnonzero(np.diff(r) != 0)
        if change.size < 2:
            continue
        runs.extend(np.diff(change).tolist())
    if not runs:
        return None
    med_n = float(np.median(runs))
    med = med_n * px
    return med if (med_n >= 3 and med >= 4 * grid_res) else None


def inspect_layer(name: str, path: str, grid: Grid, categorical: bool) -> LayerCheck:
    with rasterio.open(path) as src:
        crs = src.crs
        chk = LayerCheck(
            name=name, path=path,
            crs=crs.to_string() if crs else None,
            epsg=crs.to_epsg() if crs else None,
            res=tuple(round(r, 8) for r in src.res), shape=src.shape,
            dtype=src.dtypes[0], nodata=src.nodata, categorical=categorical)
        if crs is None:
            chk.errors.append("No coordinate reference system.")
            return chk
        gb = transform_bounds(grid.crs, crs, *grid.bounds, densify_pts=21)
        sb = src.bounds
        ix = max(0, min(gb[2], sb.right) - max(gb[0], sb.left))
        iy = max(0, min(gb[3], sb.top) - max(gb[1], sb.bottom))
        area = (gb[2] - gb[0]) * (gb[3] - gb[1])
        chk.overlap_pct = round(100 * ix * iy / area, 2) if area > 0 else 0.0
        # ground resolution in metres
        if crs.is_geographic:
            lat = (sb.top + sb.bottom) / 2
            res_m = src.res[0] * 111320 * math.cos(math.radians(lat))
        else:
            res_m = src.res[0] * (crs.linear_units_factor[1] if crs.linear_units_factor else 1)
        chk.res_ratio = round(res_m / grid.res, 3)
        # sentinel fill values hiding as data
        sample = src.read(1, out_shape=(min(src.height, 1000), min(src.width, 1000)),
                          resampling=Resampling.nearest)
        chk.fill_values_found = int(np.sum(np.abs(sample.astype(np.float64)) > FILL_ABS_LIMIT))
    if chk.overlap_pct < 99.0:
        msg = f"Covers only {chk.overlap_pct}% of the DEM extent; uncovered cells become NoData."
        (chk.errors if chk.overlap_pct < 50 else chk.warnings).append(msg)
    if chk.fill_values_found:
        chk.warnings.append("Contains fill values (|v| > 1e30) without a NoData tag; "
                            "they are treated as NoData.")
    if categorical and np.dtype(chk.dtype).kind == "f":
        chk.warnings.append("Categorical layer stored as floating point; values are rounded to class codes.")
    if chk.res_ratio > 4:
        chk.warnings.append(f"Native resolution is {chk.res_ratio:.0f}x coarser than the DEM. "
                            "Block edges can appear in the susceptibility map.")
    elif chk.res_ratio >= 0.75:
        eff = effective_resolution(path, grid.res)
        if eff:
            chk.effective_res_m = round(eff, 1)
            chk.warnings.append(f"Stored at {res_m:.0f} m but values change only every "
                                f"~{eff:.0f} m (layer was up-sampled before export). "
                                "Block edges can appear in the susceptibility map.")
    chk.resampling = choose_resampling(categorical, chk.res_ratio).name
    return chk


def choose_resampling(categorical: bool, res_ratio: float) -> Resampling:
    if categorical:
        return Resampling.mode if res_ratio < 0.75 else Resampling.nearest
    if res_ratio < 0.75:          # finer than the grid: aggregate
        return Resampling.average
    return Resampling.bilinear    # same or coarser: interpolate


def align_to_grid(path: str, grid: Grid, out_path: str, categorical=False,
                  resampling: Resampling | None = None,
                  smooth_to_m: float | None = None) -> str:
    """Re-project / resample a raster exactly onto `grid` (same CRS, transform,
    shape). NoData and fill values are propagated as NaN (or 0 for categories).

    smooth_to_m: for a continuous layer that was already up-sampled to a fine
    grid (blocky), first aggregate to this resolution and then interpolate
    bilinearly, which removes the artificial block edges.
    """
    with rasterio.open(path) as src:
        res_ratio = 1.0
        if src.crs and not src.crs.is_geographic:
            res_ratio = src.res[0] / grid.res
        rs = resampling or choose_resampling(categorical, res_ratio)
        src_nd = src.nodata
        if src_nd is None:
            src_nd = 0 if categorical and np.dtype(src.dtypes[0]).kind in "ui" else None
        with WarpedVRT(src, crs=grid.crs, transform=grid.transform,
                       width=grid.width, height=grid.height,
                       resampling=rs, src_nodata=src_nd,
                       nodata=np.nan if not categorical else 0,
                       dtype="float32") as vrt:
            arr = vrt.read(1)
    arr = clean_nodata(arr.astype(np.float32), None if not categorical else 0)
    if categorical:
        arr = np.where(np.isfinite(arr), np.rint(arr), np.nan).astype(np.float32)
        arr[arr == 0] = np.nan
    elif smooth_to_m and smooth_to_m > 2 * grid.res:
        arr = _smooth_blocky(arr, grid, smooth_to_m)
    with rasterio.open(out_path, "w", **grid.profile()) as dst:
        dst.write(arr, 1)
    return out_path


def align_tiles(paths: list[str], grid: Grid, out_path: str, categorical=False,
                resampling: Resampling | None = None, smooth_to_m: float | None = None,
                work: str | None = None) -> str:
    """Align one raster or several tiles (e.g. Google Earth Engine exports) to
    the grid. Tiles are aligned one by one and combined on the grid, so a
    large 10 m mosaic never has to be held in memory."""
    if len(paths) == 1:
        return align_to_grid(paths[0], grid, out_path, categorical, resampling, smooth_to_m)
    work = work or os.path.dirname(out_path)
    acc = None
    for i, p in enumerate(paths):
        tmp = os.path.join(work, f"_tile{i}_{os.path.basename(out_path)}")
        align_to_grid(p, grid, tmp, categorical, resampling, None)
        a = read_band(tmp)
        acc = a if acc is None else np.where(np.isfinite(acc), acc, a)
        os.remove(tmp)
    if smooth_to_m and not categorical:
        acc = _smooth_blocky(acc, grid, smooth_to_m)
    return write_array(acc, grid, out_path)


def _smooth_blocky(arr: np.ndarray, grid: Grid, coarse_m: float) -> np.ndarray:
    """Aggregate a blocky field to its true resolution, then interpolate back
    bilinearly onto the grid (mask preserved)."""
    from rasterio.warp import reproject
    f = max(2, int(round(coarse_m / grid.res)))
    ch, cw = math.ceil(grid.height / f), math.ceil(grid.width / f)
    ct = grid.transform @ rasterio.Affine.scale(f, f)
    coarse = np.full((ch, cw), np.nan, np.float32)
    reproject(arr, coarse, src_transform=grid.transform, src_crs=grid.crs,
              dst_transform=ct, dst_crs=grid.crs, resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan)
    # fill coarse NaN cells from neighbours so the edge does not shrink
    coarse = _fill_nan_nearest(coarse)
    out = np.full(grid.shape, np.nan, np.float32)
    reproject(coarse, out, src_transform=ct, src_crs=grid.crs,
              dst_transform=grid.transform, dst_crs=grid.crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    out[~np.isfinite(arr)] = np.nan
    return out


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    mask = ~np.isfinite(a)
    if not mask.any() or mask.all():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def rasterize_mask(vector_path: str, grid: Grid) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError(f"{os.path.basename(vector_path)} has no CRS (.prj missing).")
    gdf = gdf.to_crs(grid.crs)
    return rasterize(((g, 1) for g in gdf.geometry if g is not None),
                     out_shape=grid.shape, transform=grid.transform,
                     fill=0, dtype="uint8").astype(bool)


def write_array(arr: np.ndarray, grid: Grid, out_path: str, dtype="float32", nodata=np.nan):
    with rasterio.open(out_path, "w", **grid.profile(dtype=dtype, nodata=nodata)) as dst:
        dst.write(arr.astype(dtype), 1)
    return out_path
