"""Classification of susceptibility rasters into zones."""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]


def breaks(values: np.ndarray, method="jenks", k=5, sample=20000, seed=42) -> np.ndarray:
    """Upper class limits (length k). Jenks natural breaks is computed on a
    random sample of cells, as ArcGIS does for large rasters (it uses 10,000)."""
    v = values[np.isfinite(values)]
    if method == "jenks":
        import jenkspy
        rng = np.random.default_rng(seed)
        s = rng.choice(v, size=min(sample, v.size), replace=False)
        s = np.r_[s, v.min(), v.max()]
        b = np.array(jenkspy.jenks_breaks(s, n_classes=k))[1:]
    elif method == "equal":
        b = np.linspace(v.min(), v.max(), k + 1)[1:]
    elif method == "quantile":
        b = np.quantile(v, np.linspace(0, 1, k + 1)[1:])
    else:
        raise ValueError(method)
    b[-1] = v.max()
    return b


def apply(values: np.ndarray, upper: np.ndarray) -> np.ndarray:
    cls = np.searchsorted(upper, values, side="left").astype(np.float32) + 1
    cls = np.minimum(cls, len(upper))
    cls[~np.isfinite(values)] = np.nan
    return cls


def class_table(cls: np.ndarray, cell_area_m2: float, k=5):
    v = cls[np.isfinite(cls)].astype(int)
    n = np.bincount(v, minlength=k + 1)[1:]
    tot = n.sum()
    return [{"Class": CLASS_NAMES[i] if k == 5 else str(i + 1), "cells": int(n[i]),
             "pct": 100 * n[i] / tot, "km2": n[i] * cell_area_m2 / 1e6} for i in range(k)]


def landslide_density(cls: np.ndarray, ls_rc: np.ndarray, k=5):
    """Share of inventory cells in each class (a standard check of map quality)."""
    c = cls[ls_rc[:, 0], ls_rc[:, 1]]
    c = c[np.isfinite(c)].astype(int)
    n = np.bincount(c, minlength=k + 1)[1:]
    return (100 * n / max(n.sum(), 1)).tolist()
