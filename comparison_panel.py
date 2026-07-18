import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

OUTPUT_DIR = "C:\\lsm_app\\outputs"


class CompareWorker(QThread):
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    done     = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, files, test_size):
        super().__init__()
        self.files     = files
        self.test_size = test_size

    def run(self):
        import traceback
        try:
            from data.aligner         import align_raster
            from processing.factors   import derive_all_factors
            from processing.inventory import (
                load_landslide_points,
                sample_non_landslide,
                build_dataset)
            from processing.trainer   import train_model

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            aligned_dir = os.path.join(OUTPUT_DIR, "aligned")
            factors_dir = os.path.join(OUTPUT_DIR, "factors")

            # Align DEM
            self.log.emit("Aligning DEM...")
            dem_path    = self.files.get("DEM")
            os.makedirs(aligned_dir, exist_ok=True)
            dem_aligned = os.path.join(
                aligned_dir, "DEM_aligned.tif")
            align_raster(dem_path, dem_aligned)
            self.progress.emit(10)

            # Derive factors
            self.log.emit("Deriving factors...")
            optional = {
                k: v for k, v in self.files.items()
                if k not in ["DEM", "Landslide inventory",
                              "ROI boundary"]
            }
            factor_paths = derive_all_factors(
                dem_aligned, factors_dir, optional)
            self.progress.emit(25)

            # Build dataset
            self.log.emit("Building dataset...")
            shp_path = self.files.get("Landslide inventory")
            ls_coords = load_landslide_points(
                shp_path, dem_aligned)
            nls_coords = sample_non_landslide(
                dem_aligned, len(ls_coords), ls_coords)
            df = build_dataset(
                factor_paths, ls_coords, nls_coords)
            self.progress.emit(40)

            # Train all 3 models
            results = {}
            models  = ["Random Forest", "XGBoost",
                       "Logistic Regression"]
            step    = 20

            for i, model_type in enumerate(models):
                self.log.emit(
                    f"Training {model_type}...")
                model_path = os.path.join(
                    OUTPUT_DIR,
                    f"model_{model_type.replace(' ','_')}.pkl")
                (model, scaler, auc, report,
                 X_test, y_test, y_pred_prob,
                 feature_names) = train_model(
                    df, model_type,
                    self.test_size, model_path)
                results[model_type] = {
                    "auc":          auc,
                    "report":       report,
                    "y_test":       y_test,
                    "y_pred_prob":  y_pred_prob,
                }
                self.log.emit(
                    f"  {model_type}: AUC = {auc:.4f}")
                self.progress.emit(40 + step * (i + 1))

            self.progress.emit(100)
            best = max(results, key=lambda k: results[k]["auc"])
            self.log.emit("─" * 40)
            self.log.emit(
                f"Best model: {best} "
                f"(AUC = {results[best]['auc']:.4f})")
            self.done.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


class ComparisonPanel(QWidget):
    def __init__(self, upload_panel):
        super().__init__()
        self.upload_panel = upload_panel
        self.results      = {}
        layout = QVBoxLayout()

        # Header
        header = QLabel(
            "Compare all 3 models on your data "
            "and find the best one automatically.")
        header.setStyleSheet(
            "color:#aaa; font-size:12px; padding:4px;")
        layout.addWidget(header)

        # Run button
        self.run_btn = QPushButton(
            "Run all models and compare")
        self.run_btn.setFixedHeight(44)
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        self.log_box.setPlaceholderText("Log appears here...")
        layout.addWidget(self.log_box)

        # Chart area
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout()
        self.chart_widget.setLayout(self.chart_layout)
        layout.addWidget(self.chart_widget)

        # Placeholder
        self.placeholder = QLabel(
            "Run comparison to see charts here.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(
            "color:#555; font-size:13px; padding:30px;")
        self.chart_layout.addWidget(self.placeholder)

        layout.addStretch()
        self.setLayout(layout)

    def _run(self):
        files    = self.upload_panel.get_files()
        required = ["DEM", "Landslide inventory",
                    "ROI boundary"]
        missing  = [r for r in required
                    if r not in files]
        if missing:
            self.log_box.append(
                f"Missing: {', '.join(missing)}")
            return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_box.clear()
        self.log_box.append("Starting model comparison...")

        self.worker = CompareWorker(files, 0.3)
        self.worker.log.connect(self.log_box.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, results):
        self.results = results
        self.run_btn.setEnabled(True)
        self._draw_charts(results)

    def _on_error(self, msg):
        self.log_box.append("\nERROR:\n" + msg)
        self.run_btn.setEnabled(True)

    def _draw_charts(self, results):
        # Clear previous charts
        for i in reversed(range(self.chart_layout.count())):
            w = self.chart_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        models = list(results.keys())
        aucs   = [results[m]["auc"] for m in models]
        best   = models[aucs.index(max(aucs))]
        colors = [
            "#2a82da" if m == best else "#4a4a6a"
            for m in models
        ]

        # ── AUC bar chart ──────────────────────────────────
        fig1, ax1 = plt.subplots(
            figsize=(7, 3.5), facecolor="#1a1a1a")
        ax1.set_facecolor("#1a1a1a")
        bars = ax1.bar(models, aucs,
                       color=colors, edgecolor="#1a1a1a",
                       width=0.5)
        for bar, auc in zip(bars, aucs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{auc:.4f}",
                ha="center", va="bottom",
                color="white", fontsize=11,
                fontweight="bold")

        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("AUC-ROC Score", color="white")
        ax1.set_title(
            f"Model Comparison  —  Best: {best}",
            color="white", fontsize=12)
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#444")

        # Star on best bar
        best_idx = models.index(best)
        ax1.text(
            best_idx, aucs[best_idx] + 0.02,
            "★ Best", ha="center",
            color="#FFD700", fontsize=10,
            fontweight="bold")

        plt.tight_layout()
        canvas1 = FigureCanvasQTAgg(fig1)
        canvas1.setFixedHeight(280)
        self.chart_layout.addWidget(canvas1)

        # ── ROC curves for all models ──────────────────────
        from sklearn.metrics import roc_curve
        fig2, ax2 = plt.subplots(
            figsize=(7, 4), facecolor="#1a1a1a")
        ax2.set_facecolor("#1a1a1a")

        line_colors = ["#2a82da", "#e86a1a", "#2ada82"]
        for i, m in enumerate(models):
            fpr, tpr, _ = roc_curve(
                results[m]["y_test"],
                results[m]["y_pred_prob"])
            ax2.plot(
                fpr, tpr,
                color=line_colors[i], lw=2,
                label=f"{m} (AUC={results[m]['auc']:.3f})")

        ax2.plot([0, 1], [0, 1], color="#555",
                 lw=1, linestyle="--")
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.02])
        ax2.set_xlabel("False Positive Rate",
                       color="white")
        ax2.set_ylabel("True Positive Rate",
                       color="white")
        ax2.set_title("ROC Curves — All Models",
                      color="white", fontsize=12)
        ax2.tick_params(colors="white")
        ax2.legend(
            facecolor="#2a2a2a", edgecolor="#444",
            labelcolor="white", fontsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout()
        canvas2 = FigureCanvasQTAgg(fig2)
        canvas2.setFixedHeight(300)
        self.chart_layout.addWidget(canvas2)

        # Save comparison chart
        fig1.savefig(
            os.path.join(OUTPUT_DIR, "model_comparison.png"),
            dpi=150, facecolor="#1a1a1a",
            bbox_inches="tight")
        plt.close("all")