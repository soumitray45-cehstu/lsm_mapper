import os
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QGroupBox, QHBoxLayout, QSlider, QPushButton,
    QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

OUTPUT_DIR = "C:\\lsm_app\\outputs"


class WorkerThread(QThread):
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    done     = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, files, model_type,
                 test_size, method_type="Machine Learning"):
        super().__init__()
        self.files       = files
        self.model_type  = model_type
        self.test_size   = test_size
        self.method_type = method_type

    def run(self):
        try:
            from data.aligner         import align_raster
            from processing.factors   import derive_all_factors
            from processing.inventory import (
                load_landslide_points,
                sample_non_landslide,
                build_dataset)
            from processing.trainer   import train_model
            from processing.predictor import predict_lsm
            from output.visualizer    import (
                plot_lsm, make_folium_map,
                plot_roc_curve, plot_shap_importance)
            from output.exporter      import export_results

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            aligned_dir = os.path.join(OUTPUT_DIR, "aligned")
            factors_dir = os.path.join(OUTPUT_DIR, "factors")
            model_path  = os.path.join(OUTPUT_DIR, "lsm_model.pkl")
            lsm_path    = os.path.join(OUTPUT_DIR, "LSM.tif")
            png_path    = os.path.join(OUTPUT_DIR, "LSM_map.png")
            html_path   = os.path.join(OUTPUT_DIR, "LSM_map.html")
            roc_path    = os.path.join(OUTPUT_DIR, "roc_curve.png")
            shap_path   = os.path.join(OUTPUT_DIR, "shap_importance.png")

            # Step 1 — Align DEM
            self.log.emit("Aligning DEM...")
            dem_path    = self.files.get("DEM")
            os.makedirs(aligned_dir, exist_ok=True)
            dem_aligned = os.path.join(aligned_dir, "DEM_aligned.tif")
            align_raster(dem_path, dem_aligned)
            self.progress.emit(12)

            # Step 2 — Derive factors
            self.log.emit("Deriving conditioning factors from DEM...")
            optional = {
                k: v for k, v in self.files.items()
                if k not in ["DEM", "Landslide inventory",
                              "ROI boundary"]
            }
            factor_paths = derive_all_factors(
                dem_aligned, factors_dir, optional)
            self.progress.emit(30)

            # Step 3 — Build dataset
            self.log.emit("Building dataset from landslide inventory...")
            shp_path = self.files.get("Landslide inventory")
            landslide_coords = load_landslide_points(
                shp_path, dem_aligned)
            non_slide_coords = sample_non_landslide(
                dem_aligned, len(landslide_coords),
                landslide_coords)
            df = build_dataset(
                factor_paths, landslide_coords,
                non_slide_coords)
            self.progress.emit(50)

            # Step 4-6 — Train model
            if self.method_type == "Machine Learning":
                self.log.emit(
                    f"Training {self.model_type} model...")
                (model, scaler, auc, report,
                 X_test, y_test, y_pred_prob,
                 feature_names) = train_model(
                    df, self.model_type,
                    self.test_size, model_path)
                self.log.emit(
                    f"Training complete!  AUC-ROC: {auc:.4f}")
                self.progress.emit(65)

                self.log.emit("Generating ROC curve...")
                plot_roc_curve(
                    y_test, y_pred_prob, auc, roc_path)
                self.progress.emit(72)

                self.log.emit(
                    "Computing SHAP feature importance...")
                plot_shap_importance(
                    model, X_test, feature_names,
                    shap_path, self.model_type)
                self.progress.emit(80)

            else:
                from processing.trainer import run_statistical_method
                self.log.emit(
                    f"Running {self.model_type} "
                    f"statistical method...")
                scores, auc = run_statistical_method(
                    df, self.model_type)

                report = (
                    f"Method: {self.model_type}\n"
                    f"AUC-ROC: {auc:.4f}\n"
                    f"Samples: {len(df)}\n"
                    f"Landslide points: "
                    f"{int(df['label'].sum())}\n"
                    f"Non-landslide points: "
                    f"{int((df['label']==0).sum())}\n")

                self.log.emit(
                    f"Done!  AUC-ROC: {auc:.4f}")

                import joblib
                joblib.dump({
                    "scores": scores,
                    "method": self.model_type,
                    "feature_names": list(
                        df.drop(columns=["label"]).columns),
                    "df_index": df.index.tolist()
                }, model_path)

                self.progress.emit(80)

            # Step 7 — Predict
            self.log.emit("Predicting LSM raster...")
            predict_lsm(factor_paths, model_path, lsm_path)
            self.progress.emit(88)

            # Step 8 — Export
            self.log.emit("Generating map and outputs...")
            plot_lsm(lsm_path, png_path)
            make_folium_map(lsm_path, html_path)
            export_results(lsm_path, report, auc, OUTPUT_DIR)
            self.progress.emit(100)

            self.done.emit(html_path)

        except Exception:
            self.error.emit(traceback.format_exc())


class ConfigPanel(QWidget):
    def __init__(self, upload_panel, map_viewer,
                 results_panel=None):
        super().__init__()
        self.upload_panel  = upload_panel
        self.map_viewer    = map_viewer
        self.results_panel = results_panel

        layout = QVBoxLayout()

        # ── Model selector ──────────────────────────────────────
        model_group  = QGroupBox("ML model")
        model_layout = QVBoxLayout()

        self.method_type = QComboBox()
        self.method_type.addItems([
            "Machine Learning", "Statistical"])
        self.method_type.currentTextChanged.connect(
            self._on_method_change)
        model_layout.addWidget(QLabel("Method type:"))
        model_layout.addWidget(self.method_type)

        self.ml_label = QLabel("ML Algorithm:")
        self.model_box = QComboBox()
        self.model_box.addItems([
            "Random Forest", "XGBoost",
            "Logistic Regression"])
        model_layout.addWidget(self.ml_label)
        model_layout.addWidget(self.model_box)

        self.stat_label = QLabel("Statistical method:")
        self.stat_box   = QComboBox()
        self.stat_box.addItems([
            "Frequency Ratio",
            "Weights of Evidence",
            "Information Value"])
        self.stat_label.hide()
        self.stat_box.hide()
        model_layout.addWidget(self.stat_label)
        model_layout.addWidget(self.stat_box)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # ── Train / test split ──────────────────────────────────
        split_group  = QGroupBox("Train / test split")
        split_layout = QHBoxLayout()
        self.split_label  = QLabel("70% train / 30% test")
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(60, 90)
        self.split_slider.setValue(70)
        self.split_slider.valueChanged.connect(
            lambda v: self.split_label.setText(
                f"{v}% train / {100-v}% test"))
        split_layout.addWidget(self.split_slider)
        split_layout.addWidget(self.split_label)
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # ── Run button ──────────────────────────────────────────
        self.run_btn = QPushButton(
            "Train model and generate LSM")
        self.run_btn.setFixedHeight(40)
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        # ── Progress bar ────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # ── Log box ─────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "background:#1a1a1a; color:#ddd; "
            "font-family: Consolas, monospace; font-size:11px;")
        self.log_box.setFixedHeight(160)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def _on_method_change(self, method):
        if method == "Machine Learning":
            self.ml_label.show()
            self.model_box.show()
            self.stat_label.hide()
            self.stat_box.hide()
        else:
            self.ml_label.hide()
            self.model_box.hide()
            self.stat_label.show()
            self.stat_box.show()

    def _run(self):
        files = self.upload_panel.get_files()

        # Basic validation — make sure DEM and inventory are uploaded
        if not files.get("DEM"):
            self.log_box.append("❌ Please upload a DEM file first.")
            return
        if not files.get("Landslide inventory"):
            self.log_box.append("❌ Please upload a Landslide inventory file first.")
            return

        method_type = self.method_type.currentText()
        model = (self.model_box.currentText()
                 if method_type == "Machine Learning"
                 else self.stat_box.currentText())
        test_size = 1 - (self.split_slider.value() / 100)

        # Reset UI state
        self.progress.setValue(0)
        self.log_box.clear()
        self.log_box.append(f"Starting pipeline — {method_type}: {model}")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")

        # Create worker and connect ALL signals  ← THIS WAS THE BUG
        self.worker = WorkerThread(files, model, test_size, method_type)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_box.append)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)

        self.worker.start()

    def _on_done(self, html_path):
        """Called when the worker thread finishes successfully."""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Train model and generate LSM")
        self.log_box.append("✅ Pipeline complete! Loading results...")

        # Load map into map viewer tab
        self.map_viewer.load_map(html_path)

        # Load ROC curve, SHAP chart and accuracy report into results panel
        if self.results_panel:
            self.results_panel.show_results(OUTPUT_DIR)

        self.log_box.append("✅ All done! Check the Map and Results tabs.")

    def _on_error(self, error_text):
        """Called when the worker thread crashes."""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Train model and generate LSM")
        self.progress.setValue(0)
        self.log_box.append(f"\n❌ ERROR — pipeline failed:\n{error_text}")