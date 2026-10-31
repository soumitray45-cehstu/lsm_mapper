"""Validation: hold-out, repeated stratified k-fold and spatial block CV,
optional nested hyper-parameter tuning, metrics, importance, multicollinearity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV,
                                     RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from .models import SEARCH_SPACES, STAT_METHODS


def metrics(y, p, threshold=0.5, thresholded=True) -> dict:
    out = {"AUC": float(roc_auc_score(y, p))}
    if thresholded:
        yh = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        out.update(Accuracy=accuracy_score(y, yh), Sensitivity=recall_score(y, yh, zero_division=0),
                   Specificity=tn / max(tn + fp, 1), Precision=precision_score(y, yh, zero_division=0),
                   F1=f1_score(y, yh, zero_division=0), Kappa=cohen_kappa_score(y, yh),
                   TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def spatial_blocks(xy: np.ndarray, block_m: float) -> np.ndarray:
    bx = np.floor(xy[:, 0] / block_m).astype(np.int64)
    by = np.floor(xy[:, 1] / block_m).astype(np.int64)
    return bx * 1_000_003 + by


def holdout(X, y, test_size=0.3, seed=42):
    return train_test_split(np.arange(len(y)), test_size=test_size,
                            random_state=seed, stratify=y)


def tuner(pipe, method, search="random", n_iter=40, inner_folds=5, seed=42,
          space=None, n_jobs=-1, groups=None):
    space = space or SEARCH_SPACES[method]
    cv = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    if search == "grid":
        return GridSearchCV(pipe, space, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=True)
    n_comb = int(np.prod([len(v) for v in space.values()]))
    return RandomizedSearchCV(pipe, space, n_iter=min(n_iter, n_comb), scoring="roc_auc",
                              cv=cv, random_state=seed, n_jobs=n_jobs, refit=True)


def cross_validate(make, method, X, y, scheme="kfold", k=5, repeats=10, block_m=None,
                   xy=None, tune=False, tune_kw=None, seed=42, progress=None):
    """Return a DataFrame with one row of metrics per outer fold.

    When tune=True the search runs inside every outer training fold (nested
    CV), so the reported scores never see the tuning data.
    """
    thresholded = method not in STAT_METHODS
    rows = []
    if scheme == "kfold":
        splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
        splits = list(splitter.split(X, y))
    elif scheme == "spatial":
        groups = spatial_blocks(xy, block_m)
        splits = []
        for r in range(repeats):
            sg = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
            splits += list(sg.split(X, y, groups))
    else:
        raise ValueError(scheme)
    for i, (tr, te) in enumerate(splits):
        pipe = make()
        if tune:
            pipe = tuner(pipe, method, seed=seed + i, **(tune_kw or {}))
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        m = metrics(y[te], p, thresholded=thresholded)
        m.update(fold=i, repeat=i // k, n_test=int(len(te)))
        rows.append(m)
        if progress:
            progress(i + 1, len(splits))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    cols = [c for c in ["AUC", "Accuracy", "Sensitivity", "Specificity", "Precision",
                        "F1", "Kappa"] if c in df]
    return {c: {"mean": float(df[c].mean()), "sd": float(df[c].std(ddof=1))} for c in cols}


def importance(pipe, X_test, y_test, names, method, seed=42, repeats=30) -> pd.DataFrame:
    """Model-specific importance (RF: mean decrease in impurity; XGBoost:
    gain) and model-agnostic permutation importance (drop in test AUC)."""
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe.best_estimator_.named_steps["clf"]
    est = pipe if hasattr(pipe, "named_steps") else pipe.best_estimator_
    out = pd.DataFrame({"Factor": names})
    if method == "Random Forest":
        out["Impurity"] = clf.feature_importances_
    elif method == "XGBoost":
        g = clf.get_booster().get_score(importance_type="gain")
        gains = np.array([g.get(f"f{i}", 0.0) for i in range(len(names))])
        out["Gain"] = gains / gains.sum() if gains.sum() else gains
    pi = permutation_importance(est, X_test, y_test, scoring="roc_auc",
                                n_repeats=repeats, random_state=seed, n_jobs=-1)
    out["Permutation_mean"] = pi.importances_mean
    out["Permutation_sd"] = pi.importances_std
    return out.sort_values("Permutation_mean", ascending=False).reset_index(drop=True)


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor of each column (ordinary least squares)."""
    X = df.to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    out = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        Xo = np.c_[np.ones(len(X)), np.delete(X, j, 1)]
        beta, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
        r2 = 1 - ((yj - Xo @ beta) ** 2).sum() / ((yj - yj.mean()) ** 2).sum()
        out.append(1 / max(1 - r2, 1e-12))
    return pd.DataFrame({"Factor": df.columns, "VIF": out}).sort_values("VIF", ascending=False)


def vif_screen(df: pd.DataFrame, threshold=5.0) -> tuple[list, list]:
    """Drop the factor with the highest VIF until all are below `threshold`."""
    keep, dropped = list(df.columns), []
    while len(keep) > 2:
        t = vif_table(df[keep])
        top = t.iloc[0]
        if top.VIF < threshold:
            break
        keep.remove(top.Factor)
        dropped.append((top.Factor, float(top.VIF)))
    return keep, dropped
