"""Model definitions: machine-learning classifiers and bivariate statistical
methods behind one fit / predict_proba interface, so that every method goes
through the same train/test split, cross-validation and map prediction.

All defaults and search spaces are editable from the GUI (Model settings) or
the JSON run configuration; the values actually used are written to the run
report.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ML_METHODS = ["Random Forest", "XGBoost", "Logistic Regression"]
STAT_METHODS = ["Frequency Ratio", "Weights of Evidence", "Information Value"]
ALL_METHODS = ML_METHODS + STAT_METHODS

DEFAULT_PARAMS = {
    "Random Forest": dict(n_estimators=500, max_depth=None, min_samples_leaf=1,
                          max_features="sqrt", class_weight="balanced"),
    "XGBoost": dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                    reg_lambda=1.0),
    "Logistic Regression": dict(C=1.0, class_weight="balanced"),
    "Frequency Ratio": dict(n_bins=10),
    "Weights of Evidence": dict(n_bins=10),
    "Information Value": dict(n_bins=10),
}

SEARCH_SPACES = {
    "Random Forest": {
        "clf__n_estimators": [200, 300, 500, 800],
        "clf__max_depth": [None, 6, 8, 12, 16, 24],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300, 500, 800],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4, 5, 6, 8],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__reg_lambda": [0.5, 1.0, 5.0, 10.0],
    },
    "Logistic Regression": {
        "clf__C": list(np.round(np.logspace(-3, 2, 11), 5)),
    },
    "Frequency Ratio": {"clf__n_bins": [5, 8, 10, 15, 20]},
    "Weights of Evidence": {"clf__n_bins": [5, 8, 10, 15, 20]},
    "Information Value": {"clf__n_bins": [5, 8, 10, 15, 20]},
}


class Bivariate(BaseEstimator, ClassifierMixin):
    """Frequency Ratio, Weights of Evidence or Information Value.

    Class weights are learned from the *training* landslides only. The area
    term (share of the study area in each class) comes from `area_sample`, a
    random sample of valid map cells, so no test labels are ever used.
    Continuous factors are split into `n_bins` equal-area (quantile) classes
    of the study area; categorical factors use their own classes.
    The susceptibility index is the sum of class weights over all factors.
    """

    def __init__(self, method="Frequency Ratio", n_bins=10, categorical=(),
                 feature_names=(), area_sample=None):
        self.method = method
        self.n_bins = n_bins
        self.categorical = categorical
        self.feature_names = feature_names
        self.area_sample = area_sample

    def _classes(self, j, col):
        name = self.feature_names[j]
        if name in self.categorical:
            return np.where(np.isfinite(col), col, -9999).astype(np.int64)
        return np.searchsorted(self.edges_[name], col, side="right")

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        A = self.area_sample
        self.edges_, self.tables_ = {}, {}
        n_ls = max(y.sum(), 1)
        for j, name in enumerate(self.feature_names):
            a = A[name].to_numpy(float)
            a = a[np.isfinite(a)]
            if name not in self.categorical:
                q = np.unique(np.quantile(a, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                self.edges_[name] = q
            ca = self._classes(j, a)
            cl = self._classes(j, X[y == 1, j])
            classes, area_n = np.unique(ca, return_counts=True)
            p_area = area_n / area_n.sum()
            ls_n = np.array([(cl == k).sum() for k in classes], float)
            p_ls = (ls_n + 0.5) / (n_ls + 0.5 * len(classes))   # Laplace-smoothed
            if self.method == "Frequency Ratio":
                w = p_ls / p_area
            elif self.method == "Information Value":
                w = np.log(p_ls / p_area)
            else:  # Weights of Evidence, contrast C = W+ - W-
                p_nls = np.clip(p_area, 1e-9, None)              # area-based P(B|~L)
                wp = np.log(p_ls / p_nls)
                wm = np.log(np.clip(1 - p_ls, 1e-9, None) / np.clip(1 - p_nls, 1e-9, None))
                w = wp - wm
            self.tables_[name] = pd.DataFrame({"class": classes, "area_share": p_area,
                                               "landslide_cells": ls_n, "weight": w})
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        s = np.zeros(len(X))
        for j, name in enumerate(self.feature_names):
            t = self.tables_[name]
            cls = t["class"].to_numpy()
            wts = t["weight"].to_numpy()
            k = self._classes(j, X[:, j])
            pos = np.clip(np.searchsorted(cls, k), 0, len(cls) - 1)
            hit = cls[pos] == k
            neutral = 1.0 if self.method == "Frequency Ratio" else 0.0
            s += np.where(hit, wts[pos], neutral)   # unseen class -> neutral weight
        return s

    def predict_proba(self, X):
        s = self.decision_function(X)
        # monotone squashing so the output is in [0, 1]; ranking (AUC) unchanged
        p = 1.0 / (1.0 + np.exp(-(s - self.offset_) / self.scale_)) if hasattr(self, "offset_") \
            else (s - s.min()) / (np.ptp(s) + 1e-12)
        return np.c_[1 - p, p]

    def calibrate_output(self, X_area):
        s = self.decision_function(X_area)
        self.offset_ = float(np.median(s))
        self.scale_ = float(np.std(s) + 1e-12)
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(method: str, feature_names: list, categorical: list,
               params: dict | None = None, area_sample=None, seed: int = 42,
               n_jobs: int = -1) -> Pipeline:
    p = copy.deepcopy(DEFAULT_PARAMS[method])
    p.update(params or {})
    cats = [f for f in feature_names if f in categorical]
    conts = [f for f in feature_names if f not in categorical]
    idx = {f: i for i, f in enumerate(feature_names)}
    if method == "Random Forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
        pre = "passthrough"
    elif method == "XGBoost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(random_state=seed, n_jobs=n_jobs, eval_metric="logloss",
                                tree_method="hist", **p)
        pre = "passthrough"
    elif method == "Logistic Regression":
        clf = LogisticRegression(max_iter=5000, random_state=seed, **p)
        tr = [("num", StandardScaler(), [idx[f] for f in conts])]
        if cats:
            tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), [idx[f] for f in cats]))
        pre = ColumnTransformer(tr)
    elif method in STAT_METHODS:
        clf = Bivariate(method=method, categorical=tuple(categorical),
                        feature_names=tuple(feature_names), area_sample=area_sample, **p)
        pre = "passthrough"
    else:
        raise ValueError(method)
    return Pipeline([("pre", pre), ("clf", clf)])


def params_of(pipe: Pipeline) -> dict:
    clf = pipe.named_steps["clf"]
    keep = DEFAULT_PARAMS.get(_method_of(clf), {}).keys()
    gp = clf.get_params()
    return {k: _py(gp[k]) for k in keep if k in gp}


def _method_of(clf):
    n = type(clf).__name__
    return {"RandomForestClassifier": "Random Forest", "XGBClassifier": "XGBoost",
            "LogisticRegression": "Logistic Regression"}.get(n, getattr(clf, "method", n))


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
