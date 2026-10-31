"""LSM Mapper desktop interface (PyQt5).

Tabs: 1 Data  ->  2 Settings  ->  3 Run  ->  4 Results
Everything the interface does is a RunConfig handed to core.pipeline.Run,
the same engine used by the command line, so a GUI session can be saved as
JSON and repeated exactly.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QSpinBox, QTableWidget,
                             QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
                             QDialog, QDialogButtonBox, QAction)

from lsm_mapper import __version__
from lsm_mapper.core.models import ALL_METHODS, DEFAULT_PARAMS, SEARCH_SPACES
from lsm_mapper.core.pipeline import DEM_FACTORS, Run, RunConfig

RASTER_FILTER = "Raster (*.tif *.tiff *.img *.vrt);;All files (*)"
VECTOR_FILTER = "Vector (*.shp *.gpkg *.geojson);;All files (*)"


def default_output_dir():
    return os.path.join(os.path.expanduser("~"), "LSM_Mapper_runs", "run1")


# ============================================================ data tab
class PathRow(QWidget):
    def __init__(self, label, filt, required=True, save=False):
        super().__init__()
        self.filt, self.save = filt, save
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(("required" if required else "optional") + f" - {label}")
        b = QPushButton("Browse")
        b.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(b)

    def _browse(self):
        if self.filt == "dir":
            p = QFileDialog.getExistingDirectory(self, "Choose folder")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", "", self.filt)
        if p:
            self.edit.setText(p)

    def value(self):
        return self.edit.text().strip() or None


class FactorDialog(QDialog):
    """Add any conditioning factor: one raster, several tiles, or a vector
    layer turned into a distance surface."""

    def __init__(self, parent=None, spec=None):
        super().__init__(parent)
        self.setWindowTitle("Conditioning factor")
        self.paths = list(spec["paths"]) if spec else []
        f = QFormLayout(self)
        self.name = QLineEdit(spec["name"] if spec else "")
        self.kind = QComboBox()
        self.kind.addItems(["continuous", "categorical", "distance"])
        if spec:
            self.kind.setCurrentText(spec["kind"])
        self.files = QLabel(self._files_text())
        self.files.setWordWrap(True)
        pick = QPushButton("Choose file(s)...")
        pick.clicked.connect(self._pick)
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 100000)
        self.smooth.setSuffix(" m")
        self.smooth.setSpecialValueText("off")
        self.smooth.setValue((spec or {}).get("smooth_to_m") or 0)
        self.smooth.setToolTip("For a coarse continuous layer that was exported at a fine pixel size "
                               "(e.g. 5.5 km rainfall stored at 30 m): aggregate to this resolution and "
                               "interpolate bilinearly to remove artificial block edges.")
        f.addRow("Name", self.name)
        f.addRow("Type", self.kind)
        f.addRow(pick, self.files)
        f.addRow("De-block coarse layer to", self.smooth)
        f.addRow(QLabel("<small>Select several files to merge tiles (e.g. Google Earth Engine exports). "
                        "'distance' takes a vector layer (roads, faults, rivers) and computes the "
                        "Euclidean distance to it in metres.</small>"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _files_text(self):
        return "\n".join(os.path.basename(p) for p in self.paths) or "(no file)"

    def _pick(self):
        filt = VECTOR_FILTER if self.kind.currentText() == "distance" else RASTER_FILTER
        ps, _ = QFileDialog.getOpenFileNames(self, "Choose file(s)", "", filt)
        if ps:
            self.paths = ps
            self.files.setText(self._files_text())
            if not self.name.text():
                self.name.setText(os.path.splitext(os.path.basename(ps[0]))[0].split("-")[0])

    def _ok(self):
        if not self.name.text().strip() or not self.paths:
            QMessageBox.warning(self, "Factor", "Give the factor a name and choose at least one file.")
            return
        self.accept()

    def spec(self):
        return {"name": self.name.text().strip().replace(" ", "_"), "paths": self.paths,
                "kind": self.kind.currentText(), "smooth_to_m": self.smooth.value() or None}


class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        g = QGroupBox("Required inputs")
        f = QFormLayout(g)
        self.dem = PathRow("DEM (GeoTIFF)", RASTER_FILTER)
        self.inv = PathRow("landslide inventory (points or polygons)", VECTOR_FILTER)
        self.roi = PathRow("study-area boundary (default: DEM footprint)", VECTOR_FILTER, required=False)
        f.addRow("DEM", self.dem)
        f.addRow("Landslide inventory", self.inv)
        f.addRow("Study area", self.roi)
        v.addWidget(g)

        g2 = QGroupBox("Factors derived automatically from the DEM")
        grid = QGridLayout(g2)
        self.dem_checks = {}
        for i, name in enumerate(DEM_FACTORS):
            cb = QCheckBox(name.replace("_", " "))
            cb.setChecked(True)
            self.dem_checks[name] = cb
            grid.addWidget(cb, i // 5, i % 5)
        self.stream_km2 = QDoubleSpinBox()
        self.stream_km2.setRange(0.01, 100)
        self.stream_km2.setValue(1.0)
        self.stream_km2.setSuffix(" km²")
        grid.addWidget(QLabel("Stream initiation area (for distance to drainage):"), 2, 0, 1, 3)
        grid.addWidget(self.stream_km2, 2, 3)
        v.addWidget(g2)

        g3 = QGroupBox("Additional conditioning factors (any number)")
        v3 = QVBoxLayout(g3)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Files", "De-block"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._edit)
        self.specs = []
        row = QHBoxLayout()
        for txt, fn in (("Add factor", self._add), ("Edit", self._edit), ("Remove", self._remove)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        v3.addWidget(self.table)
        v3.addLayout(row)
        v.addWidget(g3, 1)
        note = QLabel("All layers are checked and aligned to the DEM grid automatically: coordinate "
                      "system, pixel size, extent and NoData. Categorical layers use nearest/majority "
                      "resampling, continuous layers bilinear or averaging.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#999")
        v.addWidget(note)

    def _refresh(self):
        self.table.setRowCount(len(self.specs))
        for r, s in enumerate(self.specs):
            for c, val in enumerate([s["name"], s["kind"],
                                     "; ".join(os.path.basename(p) for p in s["paths"]),
                                     f"{s['smooth_to_m']:.0f} m" if s.get("smooth_to_m") else ""]):
                self.table.setItem(r, c, QTableWidgetItem(val))

    def _add(self):
        d = FactorDialog(self)
        if d.exec_():
            self.specs.append(d.spec())
            self._refresh()

    def _edit(self, *_):
        r = self.table.currentRow()
        if r < 0:
            return
        d = FactorDialog(self, self.specs[r])
        if d.exec_():
            self.specs[r] = d.spec()
            self._refresh()

    def _remove(self):
        r = self.table.currentRow()
        if r >= 0:
            self.specs.pop(r)
            self._refresh()

    def fill(self, cfg: RunConfig):
        cfg.dem, cfg.inventory, cfg.roi = self.dem.value(), self.inv.value(), self.roi.value()
        cfg.dem_factors = [k for k, cb in self.dem_checks.items() if cb.isChecked()]
        cfg.stream_area_km2 = self.stream_km2.value()
        cfg.factors = [dict(s) for s in self.specs]

    def load(self, cfg: RunConfig):
        self.dem.edit.setText(cfg.dem or "")
        self.inv.edit.setText(cfg.inventory or "")
        self.roi.edit.setText(cfg.roi or "")
        for k, cb in self.dem_checks.items():
            cb.setChecked(k in cfg.dem_factors)
        self.stream_km2.setValue(cfg.stream_area_km2)
        self.specs = [dict(s) for s in cfg.factors]
        self._refresh()


# ============================================================ settings tab
class ParamTable(QTableWidget):
    """Editable hyper-parameters for one method; values are parsed as JSON."""

    def __init__(self, method):
        super().__init__(0, 3)
        self.method = method
        self.setHorizontalHeaderLabels(["Parameter", "Value (used if tuning is off)", "Tuning search space"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reset()

    def reset(self):
        d = DEFAULT_PARAMS[self.method]
        space = SEARCH_SPACES[self.method]
        self.setRowCount(len(d))
        for r, (k, v) in enumerate(d.items()):
            it = QTableWidgetItem(k)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.setItem(r, 0, it)
            self.setItem(r, 1, QTableWidgetItem(json.dumps(v)))
            self.setItem(r, 2, QTableWidgetItem(json.dumps(space.get(f"clf__{k}", ""))))

    def params(self):
        out = {}
        for r in range(self.rowCount()):
            k = self.item(r, 0).text()
            try:
                out[k] = json.loads(self.item(r, 1).text())
            except Exception:
                out[k] = self.item(r, 1).text()
        return out

    def space(self):
        out = {}
        for r in range(self.rowCount()):
            txt = self.item(r, 2).text().strip()
            if txt and txt != '""':
                try:
                    vals = json.loads(txt)
                    if isinstance(vals, list) and vals:
                        out[f"clf__{self.item(r, 0).text()}"] = vals
                except Exception:
                    pass
        return out


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        g = QGroupBox("Methods")
        h = QHBoxLayout(g)
        self.method_checks = {}
        for m in ALL_METHODS:
            cb = QCheckBox(m)
            cb.setChecked(m in ("XGBoost", "Random Forest", "Logistic Regression", "Frequency Ratio"))
            self.method_checks[m] = cb
            h.addWidget(cb)
        v.addWidget(g)

        g = QGroupBox("Hyper-parameters (edit values; lists in the last column define the tuning search)")
        gv = QVBoxLayout(g)
        self.ptabs = QTabWidget()
        self.param_tables = {}
        for m in ALL_METHODS:
            t = ParamTable(m)
            self.param_tables[m] = t
            self.ptabs.addTab(t, m)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(lambda: self.param_tables[self.ptabs.tabText(self.ptabs.currentIndex())].reset())
        gv.addWidget(self.ptabs)
        gv.addWidget(reset, 0, Qt.AlignLeft)
        self.ptabs.setMinimumHeight(220)
        v.addWidget(g)

        g = QGroupBox("Tuning and validation")
        f = QFormLayout(g)
        self.tune = QCheckBox("Tune hyper-parameters on the training set (cross-validated search, AUC)")
        self.tune.setChecked(True)
        self.search = QComboBox()
        self.search.addItems(["random", "grid"])
        self.tunecv = QComboBox()
        self.tunecv.addItems(["random", "spatial"])
        self.tunecv.setToolTip("spatial: inner folds are spatial blocks, so parameters are chosen "
                               "for prediction into unseen areas")
        self.n_iter = QSpinBox(); self.n_iter.setRange(5, 500); self.n_iter.setValue(30)
        self.inner = QSpinBox(); self.inner.setRange(2, 10); self.inner.setValue(3)
        self.folds = QSpinBox(); self.folds.setRange(2, 20); self.folds.setValue(5)
        self.repeats = QSpinBox(); self.repeats.setRange(1, 50); self.repeats.setValue(5)
        self.block = QDoubleSpinBox(); self.block.setRange(0, 500); self.block.setValue(10); self.block.setSuffix(" km")
        self.block.setSpecialValueText("off")
        self.srepeats = QSpinBox(); self.srepeats.setRange(1, 50); self.srepeats.setValue(5)
        self.nested = QCheckBox("Nested: repeat the tuning inside every outer fold")
        self.nested.setChecked(True)
        self.test = QSpinBox(); self.test.setRange(10, 50); self.test.setValue(30); self.test.setSuffix(" % test")
        self.seed = QSpinBox(); self.seed.setRange(0, 999999); self.seed.setValue(42)
        f.addRow(self.tune)
        f.addRow("Search", self.search)
        f.addRow("Random-search iterations", self.n_iter)
        f.addRow("Inner CV folds", self.inner)
        f.addRow("Inner CV type", self.tunecv)
        f.addRow("Hold-out split", self.test)
        f.addRow("k-fold CV: folds", self.folds)
        f.addRow("k-fold CV: repeats", self.repeats)
        f.addRow("Spatial block CV: block size", self.block)
        f.addRow("Spatial block CV: repeats", self.srepeats)
        f.addRow(self.nested)
        f.addRow("Random seed", self.seed)
        v.addWidget(g)

        g = QGroupBox("Non-landslide sampling")
        f = QFormLayout(g)
        self.ratio = QDoubleSpinBox(); self.ratio.setRange(0.5, 10); self.ratio.setValue(1.0); self.ratio.setSuffix(" x landslides")
        self.region = QComboBox(); self.region.addItems(["study_area", "inventory_bbox"])
        self.minslope = QDoubleSpinBox(); self.minslope.setRange(-1, 60); self.minslope.setValue(-1); self.minslope.setSuffix(" deg")
        self.minslope.setSpecialValueText("no limit")
        self.buffer = QDoubleSpinBox(); self.buffer.setRange(0, 5000); self.buffer.setValue(100); self.buffer.setSuffix(" m")
        f.addRow("Number", self.ratio)
        f.addRow("Region", self.region)
        f.addRow("Minimum slope", self.minslope)
        warn = QLabel("Restricting non-landslide cells (to steep slopes or a sub-region) means the "
                      "models never see the excluded terrain, and they can extrapolate badly onto it.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#c9a227")
        f.addRow(warn)
        f.addRow("Distance from any landslide", self.buffer)
        v.addWidget(g)

        g = QGroupBox("Reliability of the maps")
        f = QFormLayout(g)
        self.rel = QCheckBox("Area of applicability, uncertainty and artefact checks for every map")
        self.rel.setChecked(True)
        self.relk = QSpinBox(); self.relk.setRange(3, 10); self.relk.setValue(5)
        f.addRow(self.rel)
        f.addRow("Spatial folds (ensemble size)", self.relk)
        note = QLabel("Writes a dissimilarity-index map with the area outside the model's area of "
                      "applicability, a per-cell uncertainty map, the accuracy inside and outside "
                      "that area, and a block-edge artefact index for coarse input layers.")
        note.setWordWrap(True)
        f.addRow(note)
        v.addWidget(g)

        g = QGroupBox("Factors and map")
        f = QFormLayout(g)
        self.vif = QDoubleSpinBox(); self.vif.setRange(0, 100); self.vif.setValue(0); self.vif.setSpecialValueText("keep all factors")
        self.classif = QComboBox(); self.classif.addItems(["jenks", "equal", "quantile"])
        self.ncls = QSpinBox(); self.ncls.setRange(3, 10); self.ncls.setValue(5)
        self.maps = QCheckBox("Predict susceptibility maps"); self.maps.setChecked(True)
        self.out = PathRow("output folder", "dir")
        self.out.edit.setText(default_output_dir())
        f.addRow("Drop factors with VIF above", self.vif)
        f.addRow("Classification", self.classif)
        f.addRow("Number of classes", self.ncls)
        f.addRow(self.maps)
        f.addRow("Output folder", self.out)
        v.addWidget(g)
        v.addStretch()

    def fill(self, cfg: RunConfig):
        cfg.methods = [m for m, cb in self.method_checks.items() if cb.isChecked()]
        cfg.params = {m: t.params() for m, t in self.param_tables.items()}
        cfg.search_spaces = {m: t.space() for m, t in self.param_tables.items()}
        cfg.tune = self.tune.isChecked()
        cfg.tune_search = self.search.currentText()
        cfg.tune_iter = self.n_iter.value()
        cfg.tune_inner_folds = self.inner.value()
        cfg.tune_cv = self.tunecv.currentText()
        cfg.reliability = self.rel.isChecked()
        cfg.reliability_folds = self.relk.value()
        cfg.test_size = self.test.value() / 100
        cfg.cv_folds, cfg.cv_repeats = self.folds.value(), self.repeats.value()
        cfg.spatial_block_km = self.block.value() or None
        cfg.spatial_repeats = self.srepeats.value()
        cfg.nested_cv = self.nested.isChecked()
        cfg.seed = self.seed.value()
        cfg.absence_ratio = self.ratio.value()
        cfg.absence_region = self.region.currentText()
        cfg.absence_min_slope = None if self.minslope.value() < 0 else self.minslope.value()
        cfg.absence_buffer_m = self.buffer.value()
        cfg.vif_screen = self.vif.value() or None
        cfg.classification = self.classif.currentText()
        cfg.n_classes = self.ncls.value()
        cfg.make_maps = self.maps.isChecked()
        cfg.output_dir = self.out.value() or default_output_dir()

    def load(self, cfg: RunConfig):
        for m, cb in self.method_checks.items():
            cb.setChecked(m in cfg.methods)
        for m, p in (cfg.params or {}).items():
            t = self.param_tables.get(m)
            if t:
                for r in range(t.rowCount()):
                    k = t.item(r, 0).text()
                    if k in p:
                        t.item(r, 1).setText(json.dumps(p[k]))
        self.tune.setChecked(cfg.tune)
        self.search.setCurrentText(cfg.tune_search)
        self.n_iter.setValue(cfg.tune_iter)
        self.inner.setValue(cfg.tune_inner_folds)
        self.tunecv.setCurrentText(cfg.tune_cv)
        self.rel.setChecked(cfg.reliability)
        self.relk.setValue(cfg.reliability_folds)
        self.test.setValue(int(round(cfg.test_size * 100)))
        self.folds.setValue(cfg.cv_folds)
        self.repeats.setValue(cfg.cv_repeats)
        self.block.setValue(cfg.spatial_block_km or 0)
        self.srepeats.setValue(cfg.spatial_repeats)
        self.nested.setChecked(cfg.nested_cv)
        self.seed.setValue(cfg.seed)
        self.ratio.setValue(cfg.absence_ratio)
        self.region.setCurrentText(cfg.absence_region)
        self.minslope.setValue(-1 if cfg.absence_min_slope is None else cfg.absence_min_slope)
        self.buffer.setValue(cfg.absence_buffer_m)
        self.vif.setValue(cfg.vif_screen or 0)
        self.classif.setCurrentText(cfg.classification)
        self.ncls.setValue(cfg.n_classes)
        self.maps.setChecked(cfg.make_maps)
        self.out.edit.setText(cfg.output_dir)


# ============================================================ run tab
class Worker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            stage = {"n": 0}
            stages = ["Preparing factors", "Sampling", "Multicollinearity", "Modelling", "Mapping"]

            def log(s):
                for i, st in enumerate(stages):
                    if st in s:
                        self.progress.emit(int(100 * i / len(stages)))
                self.log.emit(s)

            Run(self.cfg, log=log).run()
            self.progress.emit(100)
            self.done.emit(self.cfg.output_dir)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RunTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.btn = QPushButton("Run")
        self.btn.setFixedHeight(40)
        self.btn.clicked.connect(main.start)
        save = QPushButton("Save settings...")
        save.clicked.connect(main.save_config)
        load = QPushButton("Load settings...")
        load.clicked.connect(main.load_config)
        row.addWidget(self.btn, 2)
        row.addWidget(save)
        row.addWidget(load)
        v.addLayout(row)
        self.bar = QProgressBar()
        v.addWidget(self.bar)
        self.logbox = QTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        v.addWidget(self.logbox, 1)


# ============================================================ results tab
class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.info = QLabel("Results appear here after a run.")
        v.addWidget(self.info)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(230)
        v.addWidget(self.table)
        self.figs = QTabWidget()
        v.addWidget(self.figs, 1)
        row = QHBoxLayout()
        self.open_folder = QPushButton("Open output folder")
        self.open_map = QPushButton("Open interactive map")
        self.open_report = QPushButton("Open run report")
        for b in (self.open_folder, self.open_map, self.open_report):
            b.setEnabled(False)
            row.addWidget(b)
        v.addLayout(row)
        self.out = None
        self.open_folder.clicked.connect(lambda: self._open(self.out))
        self.open_map.clicked.connect(lambda: self._open(os.path.join(self.out, "maps", "interactive_map.html")))
        self.open_report.clicked.connect(lambda: self._open(os.path.join(self.out, "run_report.json")))

    def _open(self, p):
        if p and os.path.exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def show_run(self, out):
        self.out = out
        with open(os.path.join(out, "run_report.json")) as f:
            rep = json.load(f)
        ms = list(rep.get("methods", {}))
        cols = ["Test AUC", "Accuracy", "Kappa", "k-fold CV AUC", "Spatial CV AUC", "Train AUC",
                "Outside AOA", "Mean SD"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeaderItem(3).setToolTip("Mean and SD over the outer folds of repeated k-fold CV")
        self.table.horizontalHeaderItem(4).setToolTip("Mean and SD over the outer folds of spatial block CV")
        self.table.setRowCount(len(ms))
        self.table.setVerticalHeaderLabels(ms)
        for r, m in enumerate(ms):
            e = rep["methods"][m]
            t = e["test"]
            vals = [f"{t['AUC']:.3f}", f"{t.get('Accuracy', float('nan')):.3f}" if "Accuracy" in t else "-",
                    f"{t['Kappa']:.3f}" if "Kappa" in t else "-"]
            for k in ("cv_kfold", "cv_spatial"):
                a = e.get(k, {}).get("AUC")
                vals.append(f"{a['mean']:.3f} ± {a['sd']:.3f}" if a else "-")
            vals.append(f"{e.get('train_auc', float('nan')):.3f}")
            rl = rep.get("reliability", {}).get(m)
            vals += [f"{rl['outside_aoa_pct']:.1f} %", f"{rl['uncertainty_sd_mean']:.3f}"] if rl else ["-", "-"]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(val))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        g = rep.get("grid", {})
        self.info.setText(f"Valid area {g.get('valid_area_km2', 0):,.1f} km² · "
                          f"{rep['samples']['landslide']} landslide / {rep['samples']['non_landslide']} "
                          f"non-landslide cells · runtime {rep.get('runtime_s', 0) / 60:.1f} min")
        self.figs.clear()
        for name, title in (("maps.png", "Maps"), ("roc_curves.png", "ROC"), ("cv_auc.png", "Cross-validation"),
                            ("importance.png", "Importance"), ("pearson.png", "Correlation"),
                            ("reliability.png", "Reliability")):
            p = os.path.join(out, "figures", name)
            if os.path.exists(p):
                lab = QLabel()
                lab.setPixmap(QPixmap(p).scaledToHeight(520, Qt.SmoothTransformation))
                lab.setAlignment(Qt.AlignCenter)
                sc = QScrollArea()
                sc.setWidget(lab)
                sc.setWidgetResizable(True)
                self.figs.addTab(sc, title)
        for b in (self.open_folder, self.open_map, self.open_report):
            b.setEnabled(True)


# ============================================================ main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LSM Mapper {__version__} — Landslide Susceptibility Mapping")
        self.resize(1180, 820)
        self.tabs = QTabWidget()
        self.data = DataTab()
        self.settings = SettingsTab()
        self.runtab = RunTab(self)
        self.results = ResultsTab()
        self.tabs.addTab(self.data, "1  Data")
        self.tabs.addTab(self.settings, "2  Settings")
        self.tabs.addTab(self.runtab, "3  Run")
        self.tabs.addTab(self.results, "4  Results")
        self.setCentralWidget(self.tabs)
        m = self.menuBar().addMenu("File")
        for txt, fn in (("Load settings...", self.load_config), ("Save settings...", self.save_config),
                        ("Open previous run...", self.open_run)):
            a = QAction(txt, self)
            a.triggered.connect(fn)
            m.addAction(a)
        h = self.menuBar().addMenu("Help")
        a = QAction("About LSM Mapper", self)
        a.triggered.connect(self.about)
        h.addAction(a)

    def config(self) -> RunConfig:
        cfg = RunConfig(dem="", inventory="", output_dir="")
        self.data.fill(cfg)
        self.settings.fill(cfg)
        return cfg

    def start(self):
        cfg = self.config()
        if not cfg.dem or not cfg.inventory:
            QMessageBox.warning(self, "Missing input", "Choose a DEM and a landslide inventory first.")
            self.tabs.setCurrentIndex(0)
            return
        if not cfg.methods:
            QMessageBox.warning(self, "No method", "Select at least one method.")
            return
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.output_dir, "run_config.json"))
        self.runtab.logbox.clear()
        self.runtab.btn.setEnabled(False)
        self.tabs.setCurrentIndex(2)
        self.worker = Worker(cfg)
        self.worker.log.connect(self.runtab.logbox.append)
        self.worker.progress.connect(self.runtab.bar.setValue)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, out):
        self.runtab.btn.setEnabled(True)
        self.results.show_run(out)
        self.tabs.setCurrentIndex(3)

    def _failed(self, tb):
        self.runtab.btn.setEnabled(True)
        self.runtab.logbox.append("\nERROR\n" + tb)
        QMessageBox.critical(self, "Run failed", tb.strip().splitlines()[-1])

    def save_config(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save settings", "lsm_run.json", "JSON (*.json)")
        if p:
            self.config().save(p)

    def load_config(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if p:
            self.apply_config(RunConfig.load(p))

    def apply_config(self, cfg):
        self.data.load(cfg)
        self.settings.load(cfg)

    def open_run(self):
        d = QFileDialog.getExistingDirectory(self, "Run folder")
        if d and os.path.exists(os.path.join(d, "run_report.json")):
            self.results.show_run(d)
            self.tabs.setCurrentIndex(3)

    def about(self):
        QMessageBox.about(self, "About LSM Mapper",
                          f"<b>LSM Mapper {__version__}</b><br>Free, open-source landslide susceptibility "
                          "mapping (MIT License).<br>Source: github.com/soumitray45-cehstu/lsm_mapper")


def main(argv=None):
    from PyQt5.QtWidgets import QApplication
    from lsm_mapper.gui.theme import apply_theme
    app = QApplication(argv or sys.argv)
    apply_theme(app)
    w = MainWindow()
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        w.apply_config(RunConfig.load(sys.argv[1]))
    w.show()
    return app.exec_()
