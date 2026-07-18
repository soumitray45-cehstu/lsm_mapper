import numpy as np
import rasterio
import joblib


def predict_lsm(factor_paths, model_path, out_path):
    bundle = joblib.load(model_path)

    bands  = []
    meta   = None
    names  = []

    for name, path in factor_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            bands.append(arr.flatten())
            names.append(name)
            if meta is None:
                meta = src.meta.copy()

    X     = np.stack(bands, axis=1)
    valid = ~np.isnan(X).any(axis=1)
    proba = np.full(X.shape[0], np.nan)

    # Statistical method path
    if "scores" in bundle:
        import pandas as pd
        from processing.trainer import run_statistical_method
        cols = bundle["feature_names"]
        df   = pd.DataFrame(X[valid], columns=cols)
        df["label"] = 0
        scores, _ = run_statistical_method(
            df, bundle["method"])
        proba[valid] = scores

    # ML method path
    else:
        model  = bundle["model"]
        scaler = bundle["scaler"]
        X_valid = scaler.transform(X[valid])
        raw = model.predict_proba(X_valid)[:, 1]

        # ── Percentile stretch ──────────────────────
        # Stretches the probability range so the full
        # 0-1 spectrum is used — prevents everything
        # collapsing into Very Low
        p2  = np.percentile(raw, 2)
        p98 = np.percentile(raw, 98)
        print(f"Raw probability range: "
              f"{raw.min():.4f} - {raw.max():.4f}")
        print(f"Percentile stretch: "
              f"p2={p2:.4f}, p98={p98:.4f}")

        if p98 - p2 > 0.01:
            stretched = (raw - p2) / (p98 - p2)
            stretched = np.clip(stretched, 0, 1)
        else:
            # Fallback — use rank-based normalization
            from scipy.stats import rankdata
            stretched = rankdata(raw) / len(raw)

        proba[valid] = stretched

    h, w = meta["height"], meta["width"]
    lsm  = proba.reshape(h, w).astype(np.float32)

    # ── Print class distribution ────────────────────
    flat = lsm[~np.isnan(lsm)]
    total = len(flat)
    thresholds = [(0.0, 0.2, "Very Low"),
                  (0.2, 0.4, "Low"),
                  (0.4, 0.6, "Moderate"),
                  (0.6, 0.8, "High"),
                  (0.8, 1.0, "Very High")]
    for lo, hi, label in thresholds:
        count = np.sum((flat >= lo) & (flat < hi))
        pct   = 100 * count / max(total, 1)
        print(f"  {label:<12}: "
              f"{count:>8} px ({pct:.1f}%)")

    meta.update(dtype="float32", count=1,
                nodata=np.nan)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(lsm, 1)

    print(f"LSM raster saved to {out_path}")
    return lsm