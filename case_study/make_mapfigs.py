"""Figures 8-10 in the Figure 1 map style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import mapstyle as S

MAIN, ALT, SENS = "/home/lsm/case/main", "/home/lsm/case/main_slope3", "/home/lsm/case/sens"
F = "/home/lsm/ms/figs"
COL = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]
CM = ListedColormap(COL)
MASK = f"{MAIN}/_work/valid_mask.tif"


def panel(ax, arr, ext, title, cmap, vmin, vmax, first, last, letter):
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest", zorder=2)
    S.outline(ax, MASK)
    S.frame(ax, left=True, right=last, top=True, bottom=True)
    S.compass(ax, 0.17, 0.90)
    S.scalebar(ax, 0.04, 0.025, ticks=(0, 25, 50))
    if first:
        S.labels(ax)
    ax.set_title(f"({letter}) {title}", fontsize=8.5, pad=14)


def classes(path):
    a, ext = S.read_geo(path, categorical=True)
    a[a == 0] = np.nan
    return a, ext


# ---------------- Figure 8
ms = [("XGBoost", "XGBoost"), ("Random_Forest", "Random Forest"), ("Logistic_Regression", "Logistic Regression"),
      ("Frequency_Ratio", "Frequency Ratio")]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 4.5))
for i, (ax, (k, t)) in enumerate(zip(axes, ms)):
    a, ext = classes(f"{MAIN}/maps/LSM_{k}_classes.tif")
    panel(ax, a, ext, t, CM, 0.5, 5.5, i == 0, i == 3, "abcd"[i])
fig.legend([Patch(fc=c, ec="black", lw=0.4) for c in COL], NAMES, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
           fontsize=8, title="Susceptibility class", title_fontsize=8)
fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.13, wspace=0.42)
fig.savefig(f"{F}/maps.png", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

# ---------------- Figure 9
fig, axes = plt.subplots(1, 3, figsize=(5.9, 4.7))
a, ext = classes(f"{ALT}/maps/LSM_XGBoost_classes.tif")
panel(axes[0], a, ext, "Slope > 3°, inventory box", CM, 0.5, 5.5, True, False, "a")
a, ext = classes(f"{MAIN}/maps/LSM_XGBoost_classes.tif")
panel(axes[1], a, ext, "Whole study area", CM, 0.5, 5.5, False, False, "b")
sl, ext = S.read_geo(f"{MAIN}/factors/Slope.tif", categorical=False)
flat = np.where(np.isfinite(sl), (sl < 3).astype(float), np.nan)
panel(axes[2], flat, ext, "Slope < 3°", ListedColormap(["#e3e3e3", "#2f7fc1"]), 0, 1, False, True, "c")
fig.legend([Patch(fc=c, ec="black", lw=0.4) for c in COL] + [Patch(fc="#2f7fc1", ec="black", lw=0.4)],
           NAMES + ["Slope < 3°"], loc="lower center", ncol=6, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.05, wspace=0.42)
fig.savefig(f"{F}/sampling.png", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

# ---------------- Figure 10
fig, axes = plt.subplots(1, 4, figsize=(7.4, 5.3))
R, ext = S.read_geo(f"{SENS}/Rainfall_raw_aligned.tif", categorical=True)
A, _ = S.read_geo(f"{SENS}/LSM_XGBoost_raw_rain_probability.tif", categorical=False)
B, _ = S.read_geo(f"{MAIN}/maps/LSM_XGBoost_probability.tif", categorical=False)
spec = [(R, "Rainfall as exported", "Blues", 2600, 4100, "mm/year"),
        (A, "XGBoost, raw rainfall", "RdYlGn_r", 0, 1, "Probability"),
        (B, "XGBoost, de-blocked", "RdYlGn_r", 0, 1, "Probability"),
        (A - B, "Difference (b) − (c)", "RdBu_r", -0.3, 0.3, "Difference")]
for i, (ax, (d, t, cm, v0, v1, lab)) in enumerate(zip(axes, spec)):
    ax.imshow(d, cmap=cm, vmin=v0, vmax=v1, extent=ext, interpolation="nearest", zorder=2)
    S.outline(ax, MASK)
    S.frame(ax, left=True, right=i == 3, top=True, bottom=True)
    S.compass(ax, 0.17, 0.90)
    S.scalebar(ax, 0.04, 0.025, ticks=(0, 25, 50))
    ax.set_title(f"({'abcd'[i]}) {t}", fontsize=8.5, pad=14)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(v0, v1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.075)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(lab, fontsize=7)
fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.04, wspace=0.42)
fig.savefig(f"{F}/grid.png", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("ok")
