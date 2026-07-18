from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QGroupBox, QLineEdit,
    QScrollArea, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

REQUIRED = [
    ("DEM",                        "GeoTIFF (.tif)",  False),
    ("Landslide inventory",        "Shapefile (.shp)", True),
    ("ROI boundary",               "Shapefile (.shp)", True),
]

OPTIONAL = [
    ("Soil",                       "GeoTIFF (.tif)",  False),
    ("Geology",                    "GeoTIFF (.tif)",  False),
    ("LULC",                       "GeoTIFF (.tif)",  False),
    ("Rainfall",                   "GeoTIFF (.tif)",  False),
    ("NDVI",                       "GeoTIFF (.tif)",  False),
    ("Roads",                      "Shapefile (.shp)", True),
    ("Drainage",                   "Shapefile (.shp)", True),
    ("Faults",                     "Shapefile (.shp)", True),
]

class UploadPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.files = {}
        outer = QVBoxLayout()

        # Accuracy bar
        acc_group  = QGroupBox("Expected accuracy (more optional files = higher)")
        acc_layout = QVBoxLayout()
        self.acc_bar = QProgressBar()
        self.acc_bar.setRange(0, 100)
        self.acc_bar.setValue(0)
        self.acc_bar.setFormat("Upload required files to start")
        self.acc_label = QLabel("")
        self.acc_label.setStyleSheet("color: grey; font-size: 11px;")
        acc_layout.addWidget(self.acc_bar)
        acc_layout.addWidget(self.acc_label)
        acc_group.setLayout(acc_layout)
        outer.addWidget(acc_group)

        scroll_widget = QWidget()
        inner         = QVBoxLayout(scroll_widget)

        # Required section
        req_label = QLabel("  Required files")
        req_label.setStyleSheet(
            "background:#FCEBEB; color:#A32D2D; font-weight:500;"
            "padding:4px 8px; border-radius:4px;")
        inner.addWidget(req_label)

        self.rows = {}
        for name, fmt, is_shp in REQUIRED:
            box, path_box = self._make_row(name, fmt, is_shp, required=True)
            inner.addWidget(box)
            self.rows[name] = path_box

        # Optional section
        opt_label = QLabel("  Optional files  —  each one improves accuracy")
        opt_label.setStyleSheet(
            "background:#EAF3DE; color:#3B6D11; font-weight:500;"
            "padding:4px 8px; border-radius:4px; margin-top:8px;")
        inner.addWidget(opt_label)

        for name, fmt, is_shp in OPTIONAL:
            box, path_box = self._make_row(name, fmt, is_shp, required=False)
            inner.addWidget(box)
            self.rows[name] = path_box

        inner.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        self.setLayout(outer)

    def _make_row(self, name, fmt, is_shp, required):
        group    = QGroupBox(name)
        row      = QHBoxLayout()
        path_box = QLineEdit()
        path_box.setPlaceholderText(f"{fmt}  —  {'required' if required else 'optional'}")
        path_box.setReadOnly(True)

        btn = QPushButton("Browse")
        btn.setFixedWidth(80)
        btn.clicked.connect(
            lambda _, n=name, p=path_box, s=is_shp: self._browse(n, p, s))

        clear_btn = QPushButton("✕")
        clear_btn.setFixedWidth(28)
        clear_btn.clicked.connect(
            lambda _, n=name, p=path_box: self._clear(n, p))

        row.addWidget(path_box)
        row.addWidget(btn)
        row.addWidget(clear_btn)
        group.setLayout(row)
        return group, path_box

    def _browse(self, name, path_box, is_shp):
        if is_shp:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {name}", "", "Shapefiles (*.shp)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {name}", "",
                "GeoTIFF (*.tif *.tiff)")
        if path:
            path_box.setText(path)
            self.files[name] = path
            self._update_accuracy()

    def _clear(self, name, path_box):
        path_box.clear()
        self.files.pop(name, None)
        self._update_accuracy()

    def _update_accuracy(self):
        has_req  = all(r[0] in self.files for r in REQUIRED)
        n_opt    = sum(1 for o in OPTIONAL if o[0] in self.files)
        n_opt_total = len(OPTIONAL)

        if not has_req:
            self.acc_bar.setValue(0)
            self.acc_bar.setFormat("Upload the 3 required files first")
            self.acc_label.setText("")
            return

        base  = 60
        bonus = int((n_opt / n_opt_total) * 40)
        score = base + bonus

        self.acc_bar.setValue(score)

        if score < 65:
            level = "Basic — DEM-derived factors only"
            self.acc_bar.setStyleSheet("QProgressBar::chunk { background: #E24B4A; }")
        elif score < 75:
            level = "Moderate — add Soil / Geology for better results"
            self.acc_bar.setStyleSheet("QProgressBar::chunk { background: #EF9F27; }")
        elif score < 88:
            level = "Good — add Roads / Drainage to improve further"
            self.acc_bar.setStyleSheet("QProgressBar::chunk { background: #639922; }")
        else:
            level = "Excellent — all factors loaded!"
            self.acc_bar.setStyleSheet("QProgressBar::chunk { background: #1D9E75; }")

        self.acc_bar.setFormat(f"{score}%  —  {level}")
        self.acc_label.setText(
            f"Auto-derived from DEM: Slope, Aspect, Curvature, TWI, SPI, "
            f"Plan/Profile Curvature, Elevation   |   "
            f"Optional loaded: {n_opt}/{n_opt_total}")

    def get_files(self):
        return self.files