from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel,
    QPushButton, QHBoxLayout
)
from PyQt5.QtCore import Qt


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About LSM Mapper")
        self.setFixedSize(480, 420)
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Title
        title = QLabel("LSM Mapper")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:bold; "
            "color:#2a82da; padding:10px;")
        layout.addWidget(title)

        # Subtitle
        sub = QLabel(
            "Landslide Susceptibility Mapping Tool")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(
            "font-size:13px; color:#aaa;")
        layout.addWidget(sub)

        # Divider
        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet(
            "background:#444; margin:4px 0;")
        layout.addWidget(line)

        # Info
        info = QLabel("""
<b>Version:</b> 1.0.0<br><br>
<b>Methods supported:</b><br>
&nbsp;&nbsp;Machine Learning: Random Forest, XGBoost,
Logistic Regression<br>
&nbsp;&nbsp;Statistical: Frequency Ratio,
Weights of Evidence, Information Value<br><br>
<b>Auto-derived factors:</b><br>
&nbsp;&nbsp;Slope, Aspect, Curvature,
Plan/Profile Curvature, TWI, SPI, Elevation<br><br>
<b>Optional factors:</b><br>
&nbsp;&nbsp;Soil, Geology, LULC, Rainfall, NDVI,
Distance to Roads/Drainage/Faults<br><br>
<b>Output:</b><br>
&nbsp;&nbsp;5-class LSM raster (GeoTIFF),
Interactive HTML map,<br>
&nbsp;&nbsp;ROC curve, SHAP importance chart,
Accuracy report
        """)
        info.setStyleSheet(
            "font-size:12px; color:#ccc; "
            "padding:8px; line-height:1.6;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Divider
        line2 = QLabel()
        line2.setFixedHeight(1)
        line2.setStyleSheet(
            "background:#444; margin:4px 0;")
        layout.addWidget(line2)

        # Built with
        built = QLabel(
            "Built with Python · PyQt5 · "
            "scikit-learn · XGBoost · "
            "rasterio · SHAP · folium")
        built.setAlignment(Qt.AlignCenter)
        built.setStyleSheet(
            "font-size:10px; color:#666;")
        layout.addWidget(built)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setLayout(layout)