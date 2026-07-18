import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.placeholder = QLabel(
            "Run the model to see results here.\n\n"
            "This panel will show:\n"
            "  —  ROC curve\n"
            "  —  SHAP feature importance\n"
            "  —  Model accuracy metrics")
        self.placeholder.setStyleSheet(
            "color:#666; font-size:13px; padding:30px;")
        self.placeholder.setAlignment(Qt.AlignCenter)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.hide()

        layout.addWidget(self.placeholder)
        layout.addWidget(self.inner_tabs)
        self.setLayout(layout)

    def show_results(self, results_dir):
        self.placeholder.hide()
        self.inner_tabs.clear()
        self.inner_tabs.show()

        # ROC curve tab
        roc_path = os.path.join(results_dir, "roc_curve.png")
        if os.path.exists(roc_path):
            self.inner_tabs.addTab(
                self._make_image_tab(roc_path), "ROC Curve")

        # SHAP tab
        shap_path = os.path.join(results_dir, "shap_importance.png")
        if os.path.exists(shap_path):
            self.inner_tabs.addTab(
                self._make_image_tab(shap_path), "Feature Importance (SHAP)")

        # Metrics tab
        report_path = os.path.join(results_dir, "accuracy_report.txt")
        if os.path.exists(report_path):
            self.inner_tabs.addTab(
                self._make_text_tab(report_path), "Accuracy Report")

    def _make_image_tab(self, img_path):
        from PyQt5.QtWidgets import QLabel, QScrollArea
        from PyQt5.QtGui import QPixmap
        widget = QWidget()
        layout = QVBoxLayout()
        label  = QLabel()
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaledToWidth(
            900, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        widget.setLayout(layout)
        return widget

    def _make_text_tab(self, txt_path):
        from PyQt5.QtWidgets import QTextEdit
        widget   = QWidget()
        layout   = QVBoxLayout()
        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setStyleSheet(
            "background:#1a1a1a; color:#ddd; "
            "font-family: Consolas, monospace; font-size:12px;")
        with open(txt_path, "r") as f:
            text_box.setText(f.read())
        layout.addWidget(text_box)
        widget.setLayout(layout)
        return widget