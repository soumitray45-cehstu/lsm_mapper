import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import rasterio
import folium
import os

CLASSES = [
    (0.0, 0.2, "#1a9641", "Very Low"),
    (0.2, 0.4, "#a6d96a", "Low"),
    (0.4, 0.6, "#ffffbf", "Moderate"),
    (0.6, 0.8, "#fdae61", "High"),
    (0.8, 1.0, "#d7191c", "Very High"),
]


def classify_raster(data):
    """
    Classify normalised [0,1] raster into 5 susceptibility classes.
    Because predictor.py now rescales via quantile binning before
    saving, each class will contain roughly 20% of valid pixels.
    """
    classes = np.full_like(data, np.nan, dtype=np.float32)
    for i, (lo, hi, color, label) in enumerate(CLASSES):
        if i < len(CLASSES) - 1:
            mask = (data >= lo) & (data < hi)
        else:
            mask = (data >= lo) & (data <= hi)
        classes[mask] = i + 1
    return classes


def plot_lsm(lsm_path, out_png):
    with rasterio.open(lsm_path) as src:
        data   = src.read(1).astype(np.float32)
        bounds = src.bounds

    nodata = -9999.0
    data   = np.where((data == nodata) | np.isinf(data), np.nan, data)
    classes = classify_raster(data)

    colors = [c[2] for c in CLASSES]
    labels = [c[3] for c in CLASSES]
    cmap   = mcolors.ListedColormap(colors)

    fig = plt.figure(figsize=(12, 10), facecolor="#1a1a1a")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#1a1a1a")

    ax.imshow(
        classes, cmap=cmap, vmin=1, vmax=5,
        extent=[bounds.left, bounds.right,
                bounds.bottom, bounds.top],
        aspect="auto")

    ax.grid(color="#333", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.tick_params(colors="#888", labelsize=8)
    ax.set_xlabel("Longitude", color="#888", fontsize=10)
    ax.set_ylabel("Latitude",  color="#888", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Legend
    patches = [
        mpatches.Patch(facecolor=colors[i], edgecolor="#333",
                       label=labels[i])
        for i in range(5)
    ]
    legend = ax.legend(
        handles=patches, loc="lower right",
        framealpha=0.9, facecolor="#1a1a1a",
        edgecolor="#555", labelcolor="white",
        title="Susceptibility class",
        title_fontsize=10, fontsize=9,
        borderpad=0.8, handlelength=1.5)
    legend.get_title().set_color("#aaa")

    ax.set_title(
        "Landslide Susceptibility Map",
        color="white", fontsize=15,
        fontweight="bold", pad=14)

    # Per-class pixel percentage
    total = np.sum(~np.isnan(classes))
    stats_lines = []
    for i, (lo, hi, color, label) in enumerate(CLASSES):
        count = np.sum(classes == i + 1)
        pct   = 100 * count / max(total, 1)
        stats_lines.append(f"{label}: {pct:.1f}%")
    ax.text(
        0.01, 0.99, "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=8, color="#ccc",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#1a1a1a",
                  edgecolor="#444", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor="#1a1a1a",
                bbox_inches="tight")
    plt.close()
    print(f"Map image saved to {out_png}")

def make_folium_map(lsm_path, out_html):
    import base64, io
    from PIL import Image

    with rasterio.open(lsm_path) as src:
        data   = src.read(1).astype(np.float32)
        bounds = src.bounds

    # Build RGBA image
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    color_map = [
        (0.0, 0.2,  (26,  150, 65,  200)),
        (0.2, 0.4,  (166, 217, 106, 200)),
        (0.4, 0.6,  (255, 255, 191, 200)),
        (0.6, 0.8,  (253, 174, 97,  200)),
        (0.8, 1.01, (215, 25,  28,  200)),
    ]
    for lo, hi, color in color_map:
        mask = (data >= lo) & (data < hi) & ~np.isnan(data)
        rgba[mask] = color

    # Encode as base64 PNG — avoids all file path issues
    img     = Image.fromarray(rgba, mode="RGBA")
    buffer  = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    img_url = f"data:image/png;base64,{b64_str}"

    center = [
        (bounds.bottom + bounds.top)  / 2,
        (bounds.left   + bounds.right) / 2
    ]

    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles="OpenStreetMap")

    folium.TileLayer(
        "CartoDB dark_matter",
        name="Dark basemap").add_to(m)

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[bounds.bottom, bounds.left],
                [bounds.top,    bounds.right]],
        name="LSM overlay",
        opacity=0.75,
        mercator_project=False,
        zindex=1
    ).add_to(m)

    legend_html = """
    <div style="
        position: fixed; bottom: 30px; right: 30px;
        background: rgba(10,10,10,0.85);
        padding: 14px 18px; border-radius: 10px;
        border: 1px solid #555;
        font-family: Arial, sans-serif;
        font-size: 12px; color: white; z-index: 9999;">
      <div style="font-size:13px;font-weight:bold;
                  margin-bottom:10px;color:#2a82da;">
        Susceptibility
      </div>
      <div style="display:flex;align-items:center;margin-bottom:6px;">
        <div style="width:20px;height:12px;background:#d7191c;
                    border-radius:2px;margin-right:8px;"></div>Very High
      </div>
      <div style="display:flex;align-items:center;margin-bottom:6px;">
        <div style="width:20px;height:12px;background:#fdae61;
                    border-radius:2px;margin-right:8px;"></div>High
      </div>
      <div style="display:flex;align-items:center;margin-bottom:6px;">
        <div style="width:20px;height:12px;background:#ffffbf;
                    border-radius:2px;margin-right:8px;"></div>Moderate
      </div>
      <div style="display:flex;align-items:center;margin-bottom:6px;">
        <div style="width:20px;height:12px;background:#a6d96a;
                    border-radius:2px;margin-right:8px;"></div>Low
      </div>
      <div style="display:flex;align-items:center;">
        <div style="width:20px;height:12px;background:#1a9641;
                    border-radius:2px;margin-right:8px;"></div>Very Low
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    title_html = """
    <div style="
        position: fixed; top: 15px; left: 50%;
        transform: translateX(-50%);
        background: rgba(10,10,10,0.85);
        padding: 8px 20px; border-radius: 8px;
        border: 1px solid #555;
        font-family: Arial, sans-serif;
        font-size: 14px; font-weight: bold;
        color: white; z-index: 9999;">
      Landslide Susceptibility Map — LSM Mapper
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"Interactive map saved to {out_html}")

def plot_roc_curve(y_test, y_pred_prob, auc, out_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    ax.plot(fpr, tpr, color="#2a82da", lw=2.5,
            label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#555", lw=1,
            linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2a82da")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=11)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=11)
    ax.set_title("ROC Curve", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#444",
              labelcolor="white", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.annotate(
        f"AUC = {auc:.4f}",
        xy=(0.6, 0.3), xycoords="axes fraction",
        fontsize=13, color="#2a82da", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="#1a1a1a",
                bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {out_path}")


def plot_shap_importance(model, X_sample, feature_names,
                         out_path, model_type):
    try:
        import shap

        # ── Fix: convert to plain float numpy array ───────────────
        if hasattr(X_sample, "values"):
            X_sample = X_sample.values
        X_sample = np.array(X_sample, dtype=np.float64)

        sample = X_sample[:min(200, len(X_sample))]

        if model_type in ["Random Forest", "XGBoost"]:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        else:
            explainer   = shap.LinearExplainer(model, sample)
            shap_values = explainer.shap_values(sample)

        # Handle multi-class output (list of arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Handle 3-D array (some RF versions return shape [n, feat, classes])
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        mean_shap  = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)
        sorted_feat = [feature_names[i] for i in sorted_idx]
        sorted_vals = mean_shap[sorted_idx]

        fig, ax = plt.subplots(
            figsize=(9, max(5, len(feature_names) * 0.5)),
            facecolor="#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        norm   = plt.Normalize(sorted_vals.min(), sorted_vals.max())
        colors = plt.cm.Blues(norm(sorted_vals) * 0.6 + 0.4)

        bars = ax.barh(sorted_feat, sorted_vals,
                       color=colors, edgecolor="#1a1a1a", height=0.6)

        for bar, val in zip(bars, sorted_vals):
            ax.text(
                val + sorted_vals.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left",
                color="white", fontsize=9)

        ax.set_xlabel("Mean |SHAP value|", color="white", fontsize=11)
        ax.set_title(
            f"Feature Importance (SHAP)\nModel: {model_type}",
            color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        top = sorted_feat[-1]
        ax.text(
            0.98, 0.02, f"Most influential: {top}",
            transform=ax.transAxes,
            fontsize=9, color="#2a82da",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#1a2a3a",
                      edgecolor="#2a82da", alpha=0.8))

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor="#1a1a1a",
                    bbox_inches="tight")
        plt.close()
        print(f"SHAP chart saved to {out_path}")

    except Exception as e:
        print(f"SHAP skipped: {e}")