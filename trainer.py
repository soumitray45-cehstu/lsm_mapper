import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ── Machine Learning methods ───────────────────────────────────

def train_ml_model(df, model_type="Random Forest",
                   test_size=0.3,
                   out_path="lsm_model.pkl"):
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    feature_names = list(
        df.drop(columns=["label"]).columns)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42, stratify=y)

    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2)

    elif model_type == "XGBoost":
        ratio = (len(y) - y.sum()) / max(y.sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=ratio,
            max_depth=6,
            subsample=0.8)

    else:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced")

    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_pred_prob)
    report      = classification_report(y_test, y_pred)

    joblib.dump({
        "model":         model,
        "scaler":        scaler,
        "feature_names": feature_names
    }, out_path)

    print(f"Model: {model_type}  |  AUC: {auc:.4f}")
    return (model, scaler, auc, report,
            X_test, y_test, y_pred_prob, feature_names)


def train_model(df, model_type="Random Forest",
                test_size=0.3,
                out_path="lsm_model.pkl"):
    return train_ml_model(
        df, model_type, test_size, out_path)


# ── Statistical methods ────────────────────────────────────────

def frequency_ratio(df):
    result  = pd.Series(0.0, index=df.index)
    n_total = len(df)
    n_ls    = df["label"].sum()

    for col in df.columns:
        if col == "label":
            continue
        df["_bin"] = pd.cut(
            df[col], bins=10, labels=False,
            duplicates="drop")
        for b in df["_bin"].dropna().unique():
            mask      = df["_bin"] == b
            pct_ls    = (df.loc[mask, "label"].sum()
                         / max(n_ls, 1))
            pct_total = mask.sum() / max(n_total, 1)
            fr        = pct_ls / max(pct_total, 1e-9)
            result[mask] += fr
        df.drop(columns=["_bin"], inplace=True)

    n_factors = len(df.columns) - 1
    result   /= max(n_factors, 1)
    result    = ((result - result.min()) /
                 (result.max() - result.min() + 1e-9))
    return result.values


def weights_of_evidence(df):
    result = pd.Series(0.0, index=df.index)
    n_ls   = df["label"].sum()
    n_nls  = len(df) - n_ls

    for col in df.columns:
        if col == "label":
            continue
        df["_bin"] = pd.cut(
            df[col], bins=10, labels=False,
            duplicates="drop")
        col_w = pd.Series(0.0, index=df.index)
        for b in df["_bin"].dropna().unique():
            mask    = df["_bin"] == b
            ls_in   = df.loc[mask, "label"].sum()
            nls_in  = mask.sum() - ls_in
            p_f_l   = ls_in  / max(n_ls,  1e-9)
            p_f_nl  = nls_in / max(n_nls, 1e-9)
            w_plus  = np.log(
                max(p_f_l, 1e-9) /
                max(p_f_nl, 1e-9))
            col_w[mask] = w_plus
        result += col_w
        df.drop(columns=["_bin"], inplace=True)

    result = ((result - result.min()) /
              (result.max() - result.min() + 1e-9))
    return result.values


def information_value(df):
    result        = pd.Series(0.0, index=df.index)
    n_ls          = df["label"].sum()
    n_total       = len(df)
    total_density = n_ls / max(n_total, 1e-9)

    for col in df.columns:
        if col == "label":
            continue
        df["_bin"] = pd.cut(
            df[col], bins=10, labels=False,
            duplicates="drop")
        for b in df["_bin"].dropna().unique():
            mask    = df["_bin"] == b
            ls_in   = df.loc[mask, "label"].sum()
            n_in    = mask.sum()
            density = ls_in / max(n_in, 1e-9)
            iv      = np.log(
                max(density,       1e-9) /
                max(total_density, 1e-9))
            result[mask] += iv
        df.drop(columns=["_bin"], inplace=True)

    result = ((result - result.min()) /
              (result.max() - result.min() + 1e-9))
    return result.values


def run_statistical_method(df,
                           method="Frequency Ratio"):
    from sklearn.metrics import roc_auc_score

    if method == "Frequency Ratio":
        scores = frequency_ratio(df.copy())
    elif method == "Weights of Evidence":
        scores = weights_of_evidence(df.copy())
    elif method == "Information Value":
        scores = information_value(df.copy())
    else:
        scores = frequency_ratio(df.copy())

    y   = df["label"].values
    auc = roc_auc_score(y, scores)
    return scores, auc