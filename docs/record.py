"""Drive the LSM Mapper GUI offscreen on the demo data and capture frames for
the walkthrough video. Writes frames/NNN.png and frames/shots.json
(file, caption, seconds)."""
import json
import os
import sys
import time

sys.path.insert(0, "/home/lsm/repo/lsm_mapper")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
from lsm_mapper.gui.theme import apply_theme
apply_theme(app)
from lsm_mapper.gui.app import MainWindow, FactorDialog

D = "/home/lsm/demo/data"
OUT = "/home/lsm/demo/frames"
os.makedirs(OUT, exist_ok=True)
shots = []
w = MainWindow()
w.resize(1280, 800)
w.show()


def snap(caption, sec, overlay=None):
    app.processEvents()
    pm = w.grab()
    if overlay is not None:
        app.processEvents()
        op = overlay.grab()
        p = QPainter(pm)
        p.drawPixmap(QPoint((pm.width() - op.width()) // 2, (pm.height() - op.height()) // 2), op)
        p.end()
    f = os.path.join(OUT, f"{len(shots):03d}.png")
    pm.save(f)
    shots.append({"file": f, "caption": caption, "sec": sec})


# ---------------- Data
w.tabs.setCurrentIndex(0)
snap("1  Data — start with a DEM and a landslide inventory", 3.0)
w.data.dem.edit.setText(f"{D}/DEM.tif")
w.data.inv.edit.setText(f"{D}/Landslide_inventory.shp")
snap("The DEM can be in any coordinate system; nine terrain factors are derived from it", 4.0)
spec = {"name": "NDVI", "paths": [f"{D}/NDVI_Sentinel2.tif"], "kind": "continuous", "smooth_to_m": None}
dlg = FactorDialog(w, spec)
dlg.resize(560, 300)
dlg.show()
snap("Add any number of extra factors: continuous, categorical, tiles or vector distance layers", 4.5, dlg)
dlg.close()
specs = [spec,
         {"name": "LULC", "paths": [f"{D}/LULC_WorldCover.tif"], "kind": "categorical", "smooth_to_m": None},
         {"name": "Rainfall", "paths": [f"{D}/Rainfall_CHIRPS.tif"], "kind": "continuous", "smooth_to_m": 5500},
         {"name": "Soil", "paths": [f"{D}/Soil_OpenLandMap.tif"], "kind": "categorical", "smooth_to_m": None}]
w.data.specs = specs
w.data._refresh()
snap("Coarse layers stored at a fine pixel size (e.g. 5.5 km rainfall) can be de-blocked", 4.0)

# ---------------- Settings
w.tabs.setCurrentIndex(1)
for m in w.settings.method_checks:
    w.settings.method_checks[m].setChecked(m in ("XGBoost", "Random Forest", "Logistic Regression", "Frequency Ratio"))
snap("2  Settings — choose methods; every hyper-parameter is editable", 4.0)
w.settings.ptabs.setCurrentIndex(1)
snap("Lists in the last column define the tuning search space", 4.0)
sc = w.settings.findChild(type(w.settings.ptabs).__mro__[0].__base__.__base__) if False else None
from PyQt5.QtWidgets import QScrollArea
area = w.settings.findChildren(QScrollArea)[0]
area.verticalScrollBar().setValue(330)
snap("Tuning (random or grid search), hold-out, repeated k-fold and spatial block cross-validation", 4.5)
area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())
snap("Non-landslide sampling, optional VIF screening and the map classification (Jenks by default)", 4.5)

# fast demo settings
w.settings.n_iter.setValue(8)
w.settings.inner.setValue(3)
w.settings.repeats.setValue(2)
w.settings.srepeats.setValue(2)
w.settings.block.setValue(5)
w.settings.nested.setChecked(False)
w.settings.out.edit.setText("/home/lsm/demo/run")

# ---------------- Run
state = {"t0": None, "last": 0}


def tick():
    if state["t0"] is None:
        return
    if w.worker.isRunning():
        if time.time() - state["last"] > 1.5:
            state["last"] = time.time()
            snap("3  Run — every check and step is logged (time-lapse)", 0.35)
        QTimer.singleShot(200, tick)
    else:
        finish()


def finish():
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()
    w.tabs.setCurrentIndex(3)
    r = w.results
    snap("4  Results — hold-out and cross-validated AUC for each method", 5.0)
    names = ["Maps", "ROC", "Cross-validation", "Importance", "Correlation"]
    caps = ["Susceptibility maps classified by Jenks natural breaks",
            "ROC curves on the held-out test set",
            "AUC distributions: random k-fold vs spatial block cross-validation",
            "Importance: RF impurity, XGBoost gain and permutation importance",
            "Pearson correlation between factors (VIF is in the run report)"]
    for i in range(r.figs.count()):
        r.figs.setCurrentIndex(i)
        snap(caps[names.index(r.figs.tabText(i))] if r.figs.tabText(i) in names else r.figs.tabText(i), 4.0)
    json.dump(shots, open(os.path.join(OUT, "shots.json"), "w"), indent=1)
    app.quit()


def start():
    w.start()
    state["t0"] = time.time()
    snap("3  Run — every check and step is logged (time-lapse)", 2.0)
    QTimer.singleShot(200, tick)


QTimer.singleShot(100, start)
app.exec_()
print(len(shots), "frames")
