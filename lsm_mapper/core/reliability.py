"""Reliability diagnostics produced with every susceptibility map.

1. Area of applicability (AOA; Meyer & Pebesma 2021). A dissimilarity index
   (DI) is the distance, in the importance-weighted standardised predictor
   space, from a map cell to the nearest training sample, divided by the mean
   distance between training samples. The AOA threshold is the upper whisker
   (Q75 + 1.5 IQR) of the DI of training samples computed against the training
   samples of the *other* spatial-CV folds, so it reflects how far the model
   had to extrapolate when it was cross-validated. Cells with a larger DI lie
   outside the AOA: the predictor combination there was not represented in
   the training data, and the cross-validated accuracy does not apply.

2. Accuracy against dissimilarity. Out-of-fold predictions from spatial block
   CV are grouped by the DI of each test sample (relative to its own fold's
   training set) to show whether accuracy drops as the model extrapolates.

3. Prediction uncertainty. The k models fitted in that spatial CV (same
   hyper-parameters as the final model) each predict the whole map; the
   per-cell standard deviation of their probabilities is the uncertainty map
   (a resampling ensemble in the sense of LAND-SUITE / LSAT PM, but with
   spatially disjoint training sets).

4. Block-edge artefact index for coarse input layers: mean absolute change in
   susceptibility across the cell edges of an up-sampled layer, divided by the
   mean change across all other edges (about 1 = no visible block pattern).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .models import STAT_METHODS


# ------------------------------------------------------------------ feature space
class FeatureSpace:
    """Standardised, importance-weighted predictor space for distance-based
    dissimilarity. Continuous factors are z-scored with training statistics;
    categorical factors are one-hot coded (plus an 'unseen class' column) and
    each column carries the weight of its parent factor."""

    def __init__(self, names, categorical=(), weights=None):
        self.names = list(names)
        self.categorical = [c for c in categorical if c in self.names]
        w = np.ones(len(self.names)) if weights is None else np.asarray(weights, float)
        w = np.clip(np.nan_to_num(w), 0, None)
        if w.sum() <= 0:
            w = np.ones(len(self.names))
        self.weights = w / w.max()

    def fit(self, X):
        X = np.asarray(X, float)
        self.mu_, self.sd_, self.levels_ = {}, {}, {}
        for j, n in enumerate(self.names):
            if n in self.categorical:
                self.levels_[n] = np.unique(X[np.isfinite(X[:, j]), j])
            else:
                self.mu_[n] = float(np.nanmean(X[:, j]))
                self.sd_[n] = float(np.nanstd(X[:, j]) or 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        cols = []
        for j, n in enumerate(self.names):
            w = self.weights[j]
            if n in self.categorical:
                lv = self.levels_[n]
                oh = (X[:, j, None] == lv[None, :]).astype(np.float32)
                other = (oh.sum(1, keepdims=True) == 0).astype(np.float32)
                cols.append(w * np.c_[oh, other])
            else:
                cols.append((w * (X[:, j] - self.mu_[n]) / self.sd_[n])[:, None].astype(np.float32))
        return np.concatenate(cols, 1).astype(np.float32)


def nearest_dist(A, B, chunk=50_000):
    """Euclidean distance from each row of A to its nearest row of B (brute
    force, chunked; fast for a few thousand training samples)."""
    B = np.asarray(B, np.float32)
    bb = (B * B).sum(1)
    out = np.empty(len(A), np.float32)
    for i in range(0, len(A), chunk):
        a = np.asarray(A[i:i + chunk], np.float32)
        d2 = (a * a).sum(1)[:, None] + bb[None, :] - 2.0 * a @ B.T
        out[i:i + chunk] = np.sqrt(np.clip(d2.min(1), 0, None))
    return out


def importance_weights(entry: dict, names: list):
    """Permutation importance (drop in AUC) from the run report, clipped at
    zero; None (equal weights) if the method has no importance table."""
    imp = (entry or {}).get("importance")
    if not imp:
        return None
    d = dict(zip(imp["Factor"], imp["Permutation_mean"]))
    w = np.array([max(d.get(n, 0.0), 0.0) for n in names])
    return w if w.sum() > 0 else None


def spatial_folds(xy, y, block_m, k=5, seed=42):
    from .validation import spatial_blocks
    g = spatial_blocks(xy, block_m)
    sg = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.empty(len(y), int)
    for i, (_, te) in enumerate(sg.split(np.zeros(len(y)), y, g)):
        fold[te] = i
    return fold


class AOA:
    """Dissimilarity index and area-of-applicability threshold."""

    def __init__(self, space: FeatureSpace):
        self.space = space

    def fit(self, X, fold):
        self.space.fit(X)
        T = self.space.transform(X)
        self.train_ = T
        d = pdist(T) if len(T) <= 6000 else pdist(T[np.random.default_rng(0).choice(len(T), 6000, False)])
        self.dbar_ = float(d.mean())
        di = np.empty(len(T), np.float32)
        for f in np.unique(fold):
            m = fold == f
            di[m] = nearest_dist(T[m], T[~m]) / self.dbar_
        self.train_di_ = di
        q1, q3 = np.quantile(di, [0.25, 0.75])
        whisker = q3 + 1.5 * (q3 - q1)
        self.threshold_ = float(di[di <= whisker].max())
        return self

    def di(self, X):
        return nearest_dist(self.space.transform(X), self.train_) / self.dbar_


# ------------------------------------------------------------------ fold models
def fold_models(pipe, method, X, y, fold, area_X=None):
    """Refit the final pipeline (same hyper-parameters) on each spatial-CV
    training set; return the fitted models and out-of-fold probabilities."""
    models, oof = [], np.full(len(y), np.nan)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        m = clone(pipe).fit(X[tr], y[tr])
        if method in STAT_METHODS and area_X is not None:
            m.named_steps["clf"].calibrate_output(area_X)
        oof[te] = m.predict_proba(X[te])[:, 1]
        models.append(m)
    return models, oof


def fold_di(space_names, categorical, weights, X, fold):
    """DI of each sample relative to the training set of its own CV fold."""
    out = np.empty(len(X), np.float32)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        sp = FeatureSpace(space_names, categorical, weights).fit(X[tr])
        T = sp.transform(X[tr])
        dbar = float(pdist(T).mean()) if len(T) <= 6000 else float(
            pdist(T[np.random.default_rng(0).choice(len(T), 6000, False)]).mean())
        out[te] = nearest_dist(sp.transform(X[te]), T) / dbar
    return out


def auc_by_di(y, p, di, threshold, n_bins=3):
    """AUC of out-of-fold predictions inside / outside the AOA and by DI
    tertiles (bins with only one class are reported without AUC)."""
    def _auc(m):
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            return None
        return float(roc_auc_score(y[m], p[m]))
    def _share(m):
        return float(y[m].mean()) if m.any() else None
    rows = []
    for name, m in (("all", np.ones(len(y), bool)), ("inside AOA", di <= threshold),
                    ("outside AOA", di > threshold)):
        rows.append({"group": name, "n": int(m.sum()), "AUC": _auc(m), "landslide_share": _share(m)})
    qs = np.quantile(di, np.linspace(0, 1, n_bins + 1))
    for i in range(n_bins):
        m = (di >= qs[i]) & (di <= qs[i + 1]) if i == n_bins - 1 else (di >= qs[i]) & (di < qs[i + 1])
        rows.append({"group": f"DI {qs[i]:.3f}-{qs[i + 1]:.3f}", "n": int(m.sum()), "AUC": _auc(m),
                     "landslide_share": float(y[m].mean()) if m.any() else None})
    return rows


# ------------------------------------------------------------------ artefacts
def artefact_index(prob, layer):
    """Mean |dP| across edges where `layer` changes value / mean |dP| across
    all other edges between adjacent valid cells."""
    s = np.zeros(4)
    for ax in (0, 1):
        dp = np.abs(np.diff(prob, axis=ax))
        dl = np.diff(layer, axis=ax)
        ok = np.isfinite(dp) & np.isfinite(dl)
        edge = ok & (dl != 0)
        rest = ok & (dl == 0)
        s += [dp[edge].sum(), edge.sum(), dp[rest].sum(), rest.sum()]
    if s[1] == 0 or s[3] == 0:
        return None, 0
    return float((s[0] / s[1]) / (s[2] / s[3])), int(s[1])


def coarse_layers(layer_checks, grid_res, min_ratio=4.0):
    """Names of input layers whose native or effective resolution is at least
    `min_ratio` times coarser than the analysis grid."""
    out = []
    for c in layer_checks or []:
        eff = c.get("effective_res_m")
        ratio = c.get("res_ratio") or 0
        if (eff and eff >= min_ratio * grid_res) or ratio >= min_ratio:
            out.append(c["name"])
    return out


def class_summary(cls, values, k=5):
    """Mean and 90th percentile of `values` within each susceptibility class."""
    rows = []
    for c in range(1, k + 1):
        m = (cls == c) & np.isfinite(values)
        v = values[m]
        rows.append({"class": c, "mean": float(v.mean()) if v.size else None,
                     "p90": float(np.quantile(v, 0.9)) if v.size else None})
    return rows
