"""Figures written at the end of every run (300 dpi PNG)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap

CLASS_COLORS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def all_figures(run):
    fdir = os.path.join(run.out, "figures")
    try:
        roc_figure(run.roc, os.path.join(fdir, "roc_curves.png"))
        cv_figure(run, os.path.join(fdir, "cv_auc.png"))
        importance_figure(run, os.path.join(fdir, "importance.png"))
        corr_figure(os.path.join(run.out, "tables", "pearson.csv"), os.path.join(fdir, "pearson.png"))
        if run.cfg.make_maps:
            map_panels(run, os.path.join(fdir, "maps.png"))
            interactive_map(run, os.path.join(run.out, "maps", "interactive_map.html"))
    except Exception as e:  # figures never break a run
        run.log(f"Figure error: {e}")


def roc_figure(roc, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for m, (fpr, tpr, auc) in roc.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{m} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Random (AUC = 0.500)")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cv_figure(run, path):
    rows = []
    for m in run.report["methods"]:
        for scheme, lab in (("kfold", "Random 5-fold"), ("spatial", "Spatial block")):
            f = os.path.join(run.out, "tables", f"cv_{scheme}_{m.replace(' ', '_')}.csv")
            if os.path.exists(f):
                d = pd.read_csv(f)
                rows += [{"Method": m, "Scheme": lab, "AUC": a} for a in d.AUC]
    if not rows:
        return
    d = pd.DataFrame(rows)
    methods = list(run.report["methods"])
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    for i, (lab, col) in enumerate((("Random 5-fold", "#4c78a8"), ("Spatial block", "#f58518"))):
        data = [d[(d.Method == m) & (d.Scheme == lab)].AUC.to_numpy() for m in methods]
        pos = np.arange(len(methods)) + (i - 0.5) * w
        bp = ax.boxplot([x if len(x) else [np.nan] for x in data], positions=pos, widths=w * 0.9,
                        patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=col, alpha=0.75)
        ax.plot([], [], c=col, lw=8, alpha=0.75, label=lab)
    ax.set_xticks(np.arange(len(methods)), methods, rotation=15)
    ax.set_ylabel("AUC-ROC (outer test folds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def importance_figure(run, path):
    ms = [m for m in ("Random Forest", "XGBoost") if m in run.report["methods"]]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        imp = pd.read_csv(os.path.join(run.out, "tables", f"importance_{m.replace(' ', '_')}.csv"))
        spec = "Impurity" if m == "Random Forest" else "Gain"
        imp = imp.sort_values("Permutation_mean")
        y = np.arange(len(imp))
        ax.barh(y - 0.2, imp[spec], 0.4, color="#9ecae1",
                label="Mean decrease in impurity" if spec == "Impurity" else "Gain (normalised)")
        ax.barh(y + 0.2, imp.Permutation_mean, 0.4, xerr=imp.Permutation_sd, color="#08519c",
                label="Permutation (drop in test AUC)", error_kw={"lw": 0.8})
        ax.set_yticks(y, imp.Factor.str.replace("_", " "))
        ax.set_title(m, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.axvline(0, c="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def corr_figure(csv, path):
    c = pd.read_csv(csv, index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [s.replace("_", " ") for s in c.columns]
    ax.set_xticks(range(len(c)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(c)), labels)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def read_small(path, max_px=1600):
    with rasterio.open(path) as s:
        f = max(1, int(np.ceil(max(s.height, s.width) / max_px)))
        arr = s.read(1, out_shape=(s.height // f, s.width // f),
                     resampling=rasterio.enums.Resampling.mode)
        b = s.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


def map_panels(run, path):
    ms = list(run.fitted)
    n = len(ms)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 6.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = ListedColormap(CLASS_COLORS)
    for k, (ax, m) in enumerate(zip(axes, ms)):
        arr, ext = read_small(os.path.join(run.out, "maps", f"LSM_{m.replace(' ', '_')}_classes.tif"))
        a = np.where(arr == 0, np.nan, arr)
        ax.imshow(a, cmap=cmap, vmin=0.5, vmax=5.5, extent=ext, interpolation="nearest")
        ax.set_title(f"({chr(97 + k)}) {m}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
    for ax in axes[n:]:
        ax.axis("off")
    from matplotlib.patches import Patch
    names = ["Very Low", "Low", "Moderate", "High", "Very High"]
    fig.legend([Patch(color=c) for c in CLASS_COLORS], names, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def interactive_map(run, path, max_px=2000):
    """Leaflet map (folium) with one switchable layer per method."""
    import base64
    import io

    import folium
    from PIL import Image
    from rasterio.vrt import WarpedVRT
    first = None
    layers = []
    for m in run.fitted:
        f = os.path.join(run.out, "maps", f"LSM_{m.replace(' ', '_')}_classes.tif")
        with rasterio.open(f) as src:
            with WarpedVRT(src, crs="EPSG:4326", resampling=rasterio.enums.Resampling.nearest) as v:
                k = max(1, int(np.ceil(max(v.height, v.width) / max_px)))
                a = v.read(1, out_shape=(v.height // k, v.width // k),
                           resampling=rasterio.enums.Resampling.mode)
                b = v.bounds
        rgba = np.zeros(a.shape + (4,), np.uint8)
        for i, hexc in enumerate(CLASS_COLORS, start=1):
            rgba[a == i] = [int(hexc[j:j + 2], 16) for j in (1, 3, 5)] + [210]
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        layers.append((m, url, [[b.bottom, b.left], [b.top, b.right]]))
        first = first or b
    if not layers:
        return
    fm = folium.Map(location=[(first.bottom + first.top) / 2, (first.left + first.right) / 2],
                    zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    for i, (m, url, bounds) in enumerate(layers):
        folium.raster_layers.ImageOverlay(url, bounds=bounds, name=m, show=(i == 0)).add_to(fm)
    folium.LayerControl(collapsed=False).add_to(fm)
    legend = "".join(f'<div><span style="background:{c};width:14px;height:14px;display:inline-block;'
                     f'margin-right:6px"></span>{n}</div>' for c, n in
                     zip(CLASS_COLORS, ["Very Low", "Low", "Moderate", "High", "Very High"]))
    fm.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;'
        f'padding:8px 10px;border-radius:4px;font:12px sans-serif">{legend}</div>'))
    fm.save(path)
