"""Known-answer and robustness tests for LSM Mapper's engine.

Run:  python -m pytest -q tests
"""
import math
import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lsm_mapper.core import raster, terrain
from lsm_mapper.core.models import make_model, ALL_METHODS
from lsm_mapper.core.raster import Grid

UTM = CRS.from_epsg(32646)


def _write(path, arr, crs, transform, nodata=None, dtype=None):
    dtype = dtype or arr.dtype
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr.astype(dtype), 1)
    return path


# ---------------------------------------------------------------- terrain maths
def test_slope_aspect_on_plane():
    cell = 30.0
    r, c = np.mgrid[0:50, 0:60].astype(float)
    # rises 0.5 m per metre towards the east -> slope atan(0.5), faces west (270)
    z = 0.5 * c * cell
    d = terrain.surface_derivatives(z, cell)
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(d["Slope"][inner], math.degrees(math.atan(0.5)), atol=1e-4)
    assert np.allclose(d["Aspect"][inner], 270.0, atol=1e-4)
    assert np.allclose(d["Curvature"][inner], 0, atol=1e-6)


def test_slope_independent_of_cell_size_units():
    r, c = np.mgrid[0:40, 0:40].astype(float)
    for cell in (10.0, 30.0, 250.0):
        z = 0.2 * r * cell          # same 0.2 gradient on every grid
        s = terrain.surface_derivatives(z, cell)["Slope"][3:-3, 3:-3]
        assert np.allclose(s, math.degrees(math.atan(0.2)), atol=1e-4)


def test_curvature_sign_on_bowl_and_dome():
    cell = 10.0
    r, c = np.mgrid[-20:21, -20:21].astype(float) * cell
    bowl = (r ** 2 + c ** 2) / 1000.0
    cb = terrain.surface_derivatives(bowl, cell)["Curvature"][20, 20]
    cd = terrain.surface_derivatives(-bowl, cell)["Curvature"][20, 20]
    assert cb < 0 < cd          # ArcGIS convention: concave negative, convex positive


def test_flow_accumulation_valley():
    # V-shaped valley draining south: the channel collects the whole grid
    H, W = 60, 41
    r, c = np.mgrid[0:H, 0:W].astype(float)
    z = np.abs(c - 20) * 2.0 + (H - r) * 0.5
    acc = terrain.flow_accumulation(z)
    assert acc.max() >= 0.9 * H * W
    assert np.argmax(acc[-1]) == 20


def test_depression_is_filled_and_drained():
    z = np.full((30, 30), 10.0)
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 100.0   # walls
    z[15, 0] = 0.0                                      # single outlet
    z[10:20, 10:20] = 5.0                               # closed pit inside
    acc = terrain.flow_accumulation(z)
    assert np.isfinite(acc).all()
    assert acc[15, 0] >= 28 * 28                        # pit and plateau drain out


def test_hydrology_outputs_finite_and_positive_twi():
    rng = np.random.default_rng(0)
    r, c = np.mgrid[0:80, 0:80].astype(float)
    z = 100 - 0.3 * r + 0.1 * np.sin(c / 5) * 10 + rng.normal(0, 0.2, r.shape)
    s = terrain.surface_derivatives(z, 30.0)["Slope"]
    h = terrain.hydrology(z, s, 30.0, stream_area_km2=0.5)
    assert np.isfinite(h["TWI"]).all() and np.isfinite(h["SPI"]).all()
    assert h["Dist_Drainage"].min() == 0


# ---------------------------------------------------------------- alignment set
@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("align")
    t = from_origin(400000, 2500000, 30, 30)
    dem = np.add.outer(np.arange(200), np.arange(300)).astype("float32")
    p = _write(d / "dem.tif", dem, UTM, t)
    grid, _ = raster.reference_grid(str(p))
    return d, grid, t


def _ramp_value(x, y):
    return (x - 400000) / 1000 + (y - 2494000) / 1000


def test_geographic_10m_layer(ref):
    """NDVI-like layer in WGS84 at ~10 m, larger extent."""
    d, grid, _ = ref
    from rasterio.warp import transform_bounds
    b = transform_bounds(UTM, "EPSG:4326", *grid.bounds)
    res = 0.0001
    W = int((b[2] - b[0]) / res) + 40
    H = int((b[3] - b[1]) / res) + 40
    t = from_origin(b[0] - 20 * res, b[3] + 20 * res, res, res)
    lon = t.c + (np.arange(W) + 0.5) * res
    lat = t.f - (np.arange(H) + 0.5) * res
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    LON, LAT = np.meshgrid(lon, lat)
    X, Y = tr.transform(LON, LAT)
    arr = _ramp_value(X, Y).astype("float32")
    p = _write(d / "geo10.tif", arr, CRS.from_epsg(4326), t)
    chk = raster.inspect_layer("ndvi", str(p), grid, False)
    assert chk.overlap_pct >= 99 and chk.res_ratio < 0.75 and chk.resampling == "average"
    out = raster.align_to_grid(str(p), grid, str(d / "geo10_al.tif"))
    a = raster.read_band(out)
    assert a.shape == grid.shape and np.isfinite(a).all()
    xs = grid.transform.c + (np.arange(grid.width) + 0.5) * grid.res
    ys = grid.transform.f - (np.arange(grid.height) + 0.5) * grid.res
    truth = _ramp_value(*np.meshgrid(xs, ys))
    assert np.nanmax(np.abs(a - truth)) < 0.02          # < 20 m positional error


def test_coarse_250m_categorical_offset_extent(ref):
    """Soil-like 250 m categorical layer, shifted origin, partial cover."""
    d, grid, _ = ref
    t = from_origin(399880, 2500130, 250, 250)
    cls = np.tile(np.array([[3, 7], [7, 11]], "uint8"), (20, 20))
    p = _write(d / "soil.tif", cls, UTM, t, nodata=255)
    chk = raster.inspect_layer("soil", str(p), grid, True)
    assert chk.resampling == "nearest" and any("coarser" in w for w in chk.warnings)
    a = raster.read_band(raster.align_to_grid(str(p), grid, str(d / "soil_al.tif"), categorical=True))
    vals = set(np.unique(a[np.isfinite(a)]).astype(int))
    assert vals <= {3, 7, 11}                           # no blended class codes


def test_fill_values_and_nodata_variants(ref):
    """float32-max fill without a NoData tag, -9999 with a tag, 0 for categories."""
    d, grid, t = ref
    a = np.ones(grid.shape, "float32")
    a[:10] = 3.4028235e38
    a[-10:] = -3.4028235e38
    p = _write(d / "fill.tif", a, UTM, t)
    assert raster.inspect_layer("x", str(p), grid, False).fill_values_found > 0
    r = raster.read_band(raster.align_to_grid(str(p), grid, str(d / "fill_al.tif")))
    assert np.isnan(r[:10]).all() and np.isnan(r[-10:]).all() and np.nanmax(r) == 1
    b = np.full(grid.shape, 5.0, "float32")
    b[50:60] = -9999
    p2 = _write(d / "nd.tif", b, UTM, t, nodata=-9999)
    r2 = raster.read_band(raster.align_to_grid(str(p2), grid, str(d / "nd_al.tif"),
                                               resampling=raster.Resampling.nearest))
    assert np.isnan(r2[50:60]).all() and np.nanmin(r2) == 5


def test_upsampled_layer_detected_and_smoothed(ref):
    """A 5.5 km field exported at 30 m is flagged and can be de-blocked."""
    d, grid, t = ref
    yy, xx = np.mgrid[0:grid.height, 0:grid.width]
    blocks = ((yy // 60) * 7 + (xx // 60) * 3).astype("float32")   # 1.8 km blocks
    p = _write(d / "rain.tif", blocks, UTM, t)
    chk = raster.inspect_layer("rain", str(p), grid, False)
    assert chk.effective_res_m and chk.effective_res_m >= 1000
    sm = raster.read_band(raster.align_to_grid(str(p), grid, str(d / "rain_sm.tif"),
                                               smooth_to_m=1800))
    jumps_raw = np.abs(np.diff(blocks, axis=1)).max()
    jumps_sm = np.nanmax(np.abs(np.diff(sm, axis=1)))
    assert jumps_sm < jumps_raw / 5


def test_missing_crs_rejected(ref, tmp_path):
    d, grid, t = ref
    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32") as w:
        w.write(np.ones((10, 10), "float32"), 1)
    assert raster.inspect_layer("x", str(p), grid, False).errors


def test_low_overlap_rejected(ref):
    d, grid, _ = ref
    p = _write(d / "far.tif", np.ones((50, 50), "float32"), UTM, from_origin(900000, 2500000, 30, 30))
    assert raster.inspect_layer("x", str(p), grid, False).errors


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_method_fits_and_ranks(method):
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 400
    X = np.c_[rng.normal(size=n), rng.normal(size=n), rng.integers(1, 4, n)]
    y = (X[:, 0] + 0.5 * (X[:, 2] == 2) + rng.normal(0, 0.7, n) > 0.3).astype(int)
    area = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000),
                         "lc": rng.integers(1, 4, 5000).astype(float)})
    m = make_model(method, ["a", "b", "lc"], ["lc"], area_sample=area)
    m.fit(X[:300], y[:300])
    from sklearn.metrics import roc_auc_score
    p = m.predict_proba(X[300:])[:, 1]
    assert 0 <= p.min() and p.max() <= 1
    assert roc_auc_score(y[300:], p) > 0.7
