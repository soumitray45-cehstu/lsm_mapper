"""Terrain and hydrological factors derived from a DEM on a projected grid.

Slope and aspect: Horn (1981) 3x3 finite differences.
Curvatures: Zevenbergen & Thorne (1987) quadratic surface, expressed in the
same sign convention and units (1/100 z-units) as ArcGIS Pro.
Flow routing: Priority-Flood+epsilon depression filling (Barnes et al., 2014),
D8 flow directions (O'Callaghan & Mark, 1984), flow accumulation.
TWI = ln(a / tan b) (Beven & Kirkby, 1979); SPI = ln(1 + a tan b)
(Moore et al., 1991), with a the specific catchment area (m^2/m).
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from numba import njit

MIN_TAN = 1e-3  # tan(0.057 deg): avoids division by zero on flats

_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit(cache=True)
def _window(z, r, c, H, W):
    """3x3 neighbourhood; missing neighbours replaced by the centre value."""
    out = np.empty(9, dtype=np.float64)
    zc = z[r, c]
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not np.isnan(z[rr, cc]):
                out[k] = z[rr, cc]
            else:
                out[k] = zc
            k += 1
    return out


@njit(cache=True)
def _surface_kernel(z, cell):
    H, W = z.shape
    slope = np.full((H, W), np.nan, np.float32)
    aspect = np.full((H, W), np.nan, np.float32)
    curv = np.full((H, W), np.nan, np.float32)
    plan = np.full((H, W), np.nan, np.float32)
    prof = np.full((H, W), np.nan, np.float32)
    L = cell
    for r in range(H):
        for c in range(W):
            if np.isnan(z[r, c]):
                continue
            w = _window(z, r, c, H, W)
            a, b, cc_, d, e, f, g, h, i = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]
            # Horn
            dzdx = ((cc_ + 2 * f + i) - (a + 2 * d + g)) / (8 * L)
            dzdy = ((g + 2 * h + i) - (a + 2 * b + cc_)) / (8 * L)   # +y = south (row order)
            s = math.sqrt(dzdx * dzdx + dzdy * dzdy)
            slope[r, c] = math.degrees(math.atan(s))
            if dzdx == 0.0 and dzdy == 0.0:
                aspect[r, c] = -1.0          # flat
            else:
                asp = math.degrees(math.atan2(dzdy, -dzdx))
                # convert to compass bearing of the downslope direction (0 = north)
                if asp < 0:
                    asp = 90.0 - asp
                elif asp > 90.0:
                    asp = 360.0 - asp + 90.0
                else:
                    asp = 90.0 - asp
                aspect[r, c] = asp % 360.0
            # Zevenbergen & Thorne
            D = ((d + f) / 2 - e) / (L * L)
            E = ((b + h) / 2 - e) / (L * L)
            F = (-a + cc_ + g - i) / (4 * L * L)
            G = (-d + f) / (2 * L)
            Hh = (b - h) / (2 * L)
            curv[r, c] = -2.0 * (D + E) * 100.0
            gh = G * G + Hh * Hh
            if gh > 0:
                prof[r, c] = -2.0 * (D * G * G + E * Hh * Hh + F * G * Hh) / gh * 100.0
                plan[r, c] = 2.0 * (D * Hh * Hh + E * G * G - F * G * Hh) / gh * 100.0
            else:
                prof[r, c] = 0.0
                plan[r, c] = 0.0
    return slope, aspect, curv, plan, prof


def surface_derivatives(dem: np.ndarray, cell: float) -> dict:
    s, a, cu, pl, pr = _surface_kernel(dem.astype(np.float64), float(cell))
    return {"Slope": s, "Aspect": a, "Curvature": cu,
            "Plan_Curvature": pl, "Profile_Curvature": pr}


@njit(cache=True)
def _priority_flood_eps(z):
    """Priority-Flood+epsilon (Barnes et al. 2014). NaN cells are outside the
    domain; any valid cell bordering NaN or the array edge is an outlet."""
    H, W = z.shape
    out = z.copy()
    closed = np.zeros((H, W), np.bool_)
    pq = [(0.0, np.int64(0))]
    pq.pop()
    pit = np.empty(H * W, np.int64)
    p_head = 0
    p_tail = 0
    for r in range(H):
        for c in range(W):
            if np.isnan(z[r, c]):
                closed[r, c] = True
                continue
            edge = r == 0 or c == 0 or r == H - 1 or c == W - 1
            if not edge:
                for k in range(8):
                    if np.isnan(z[r + _DR[k], c + _DC[k]]):
                        edge = True
                        break
            if edge:
                closed[r, c] = True
                heapq.heappush(pq, (out[r, c], np.int64(r * W + c)))
    while len(pq) > 0 or p_head < p_tail:
        if p_head < p_tail:
            idx = pit[p_head]
            p_head += 1
            if p_head == p_tail:
                p_head = 0
                p_tail = 0
        else:
            _, idx = heapq.heappop(pq)
        r = idx // W
        c = idx % W
        zc = out[r, c]
        for k in range(8):
            rr = r + _DR[k]
            cc = c + _DC[k]
            if rr < 0 or cc < 0 or rr >= H or cc >= W or closed[rr, cc]:
                continue
            closed[rr, cc] = True
            nz = np.nextafter(zc, np.inf)
            if out[rr, cc] <= nz:
                out[rr, cc] = nz
                pit[p_tail] = rr * W + cc
                p_tail += 1
            else:
                heapq.heappush(pq, (out[rr, cc], np.int64(rr * W + cc)))
    return out


@njit(cache=True)
def _d8_accumulate(z):
    """D8 directions on a depression-free surface and upslope cell counts."""
    H, W = z.shape
    rec = np.full(H * W, -1, np.int64)
    indeg = np.zeros(H * W, np.int32)
    sq2 = math.sqrt(2.0)
    for r in range(H):
        for c in range(W):
            if np.isnan(z[r, c]):
                continue
            best = 0.0
            bi = -1
            for k in range(8):
                rr = r + _DR[k]
                cc = c + _DC[k]
                if rr < 0 or cc < 0 or rr >= H or cc >= W or np.isnan(z[rr, cc]):
                    continue
                dist = sq2 if (k % 2 == 1) else 1.0
                drop = (z[r, c] - z[rr, cc]) / dist
                if drop > best:
                    best = drop
                    bi = rr * W + cc
            rec[r * W + c] = bi
            if bi >= 0:
                indeg[bi] += 1
    acc = np.zeros(H * W, np.float64)
    q = np.empty(H * W, np.int64)
    head = 0
    tail = 0
    for i in range(H * W):
        if not np.isnan(z[i // W, i % W]):
            acc[i] = 1.0
            if indeg[i] == 0:
                q[tail] = i
                tail += 1
    while head < tail:
        i = q[head]
        head += 1
        j = rec[i]
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                q[tail] = j
                tail += 1
    return acc.reshape(H, W)


def flow_accumulation(dem: np.ndarray) -> np.ndarray:
    filled = _priority_flood_eps(dem.astype(np.float64))
    acc = _d8_accumulate(filled)
    acc[np.isnan(dem)] = np.nan
    return acc


def hydrology(dem: np.ndarray, slope_deg: np.ndarray, cell: float,
              stream_area_km2: float = 1.0) -> dict:
    acc = flow_accumulation(dem)
    sca = acc * cell                         # specific catchment area, m^2/m
    tanb = np.maximum(np.tan(np.radians(slope_deg.astype(np.float64))), MIN_TAN)
    twi = np.log(sca / tanb).astype(np.float32)
    spi = np.log1p(sca * tanb).astype(np.float32)
    streams = acc * cell * cell >= stream_area_km2 * 1e6
    out = {"TWI": twi, "SPI": spi,
           "Dist_Drainage": distance_to(streams, cell, valid=np.isfinite(dem))}
    for v in out.values():
        v[~np.isfinite(dem)] = np.nan
    out["_flow_acc"] = acc.astype(np.float32)
    out["_streams"] = streams
    return out


def distance_to(features: np.ndarray, cell: float, valid=None) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    if not features.any():
        raise ValueError("No features to measure distance to.")
    d = distance_transform_edt(~features, sampling=(cell, cell)).astype(np.float32)
    if valid is not None:
        d[~valid] = np.nan
    return d
