"""Landslide / non-landslide samples and the training table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from .raster import Grid


def load_inventory(path: str, grid: Grid, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return unique (row, col) cells holding at least one landslide.

    Polygons are represented by their centroid; points that fall outside the
    grid or on NoData cells are reported, not silently dropped.
    """
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("Landslide inventory has no CRS (.prj missing).")
    n_in = len(gdf)
    gdf = gdf.to_crs(grid.crs)
    geoms = [g.centroid if g.geom_type != "Point" else g for g in gdf.geometry if g is not None]
    xs = np.array([g.x for g in geoms])
    ys = np.array([g.y for g in geoms])
    rows, cols = rasterio.transform.rowcol(grid.transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (cols >= 0) & (rows < grid.height) & (cols < grid.width)
    rc = np.stack([rows[inside], cols[inside]], 1)
    on_valid = valid[rc[:, 0], rc[:, 1]]
    rc_valid = rc[on_valid]
    uniq = np.unique(rc_valid, axis=0)
    info = {"records": int(n_in), "outside_grid": int((~inside).sum()),
            "on_nodata": int((~on_valid).sum()),
            "duplicate_cells": int(len(rc_valid) - len(uniq)),
            "landslide_cells": int(len(uniq))}
    return uniq, info


def sample_absence(grid: Grid, valid: np.ndarray, ls_rc: np.ndarray, n: int,
                   seed: int = 42, slope: np.ndarray | None = None,
                   min_slope: float | None = None, buffer_m: float = 100.0,
                   region: str = "inventory_bbox", bbox_pad: float = 0.05) -> tuple[np.ndarray, dict]:
    """Random non-landslide cells.

    region='inventory_bbox' restricts to the landslide bounding box (+pad);
    'study_area' uses every valid cell. Cells within `buffer_m` of a landslide
    and (optionally) flatter than `min_slope` degrees are excluded.
    """
    from scipy.ndimage import distance_transform_edt
    cand = valid.copy()
    if region == "inventory_bbox":
        r0, c0 = ls_rc.min(0)
        r1, c1 = ls_rc.max(0)
        pr = int((r1 - r0) * bbox_pad)
        pc = int((c1 - c0) * bbox_pad)
        box = np.zeros_like(cand)
        box[max(0, r0 - pr): r1 + pr + 1, max(0, c0 - pc): c1 + pc + 1] = True
        cand &= box
    if min_slope is not None and slope is not None:
        cand &= np.nan_to_num(slope, nan=-1) > min_slope
    if buffer_m and buffer_m > 0:
        ls_mask = np.zeros(grid.shape, bool)
        ls_mask[ls_rc[:, 0], ls_rc[:, 1]] = True
        # distance only needed inside the bounding rows/cols of candidates
        d = distance_transform_edt(~ls_mask, sampling=(grid.res, grid.res))
        cand &= d > buffer_m
        del d
    idx = np.flatnonzero(cand)
    if len(idx) < n:
        raise ValueError(f"Only {len(idx)} eligible non-landslide cells for {n} requested.")
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=n, replace=False)
    rc = np.stack(np.unravel_index(pick, grid.shape), 1)
    info = {"eligible_cells": int(len(idx)), "sampled": int(n), "region": region,
            "min_slope_deg": min_slope, "buffer_m": buffer_m, "seed": seed}
    return rc, info


def build_table(factors: dict[str, str], grid: Grid, ls_rc: np.ndarray,
                nls_rc: np.ndarray) -> pd.DataFrame:
    """Factor values at sample cells, plus label and map coordinates."""
    rc = np.vstack([ls_rc, nls_rc])
    data = {}
    for name, path in factors.items():
        with rasterio.open(path) as src:
            arr = src.read(1, masked=False).astype(np.float32)
        data[name] = arr[rc[:, 0], rc[:, 1]]
        del arr
    df = pd.DataFrame(data)
    xs, ys = rasterio.transform.xy(grid.transform, rc[:, 0], rc[:, 1])
    df["x"] = np.asarray(xs)
    df["y"] = np.asarray(ys)
    df["label"] = np.r_[np.ones(len(ls_rc), int), np.zeros(len(nls_rc), int)]
    return df
