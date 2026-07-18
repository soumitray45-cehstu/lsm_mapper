from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
)
import os
import subprocess
import sys

class MapViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.html_path = None
        layout = QVBoxLayout()

        # Placeholder message
        self.status_label = QLabel(
            "Run the model to generate your LSM map.\n"
            "The map will be ready to open here when done.")
        self.status_label.setStyleSheet(
            "color:#666; font-size:13px; padding:20px;")

        # Open in browser button
        self.open_btn = QPushButton("Open interactive map in browser")
        self.open_btn.setFixedHeight(40)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_browser)

        # Open outputs folder button
        self.folder_btn = QPushButton("Open outputs folder")
        self.folder_btn.setFixedHeight(40)
        self.folder_btn.clicked.connect(self._open_folder)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.folder_btn)

        # Info box
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "color:#aaa; font-size:12px; padding:10px;")
        self.info_label.setWordWrap(True)

        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addLayout(btn_row)
        layout.addWidget(self.info_label)
        layout.addStretch()
        self.setLayout(layout)

    def load_map(self, html_path):
        self.html_path = html_path
        self.open_btn.setEnabled(True)
        self.status_label.setText(
            "LSM map generated successfully!")
        self.status_label.setStyleSheet(
            "color:#2a82da; font-size:16px; "
            "font-weight:bold; padding:20px;")
        self.info_label.setText(
            f"Map saved to:\n{html_path}\n\n"
            f"Click 'Open interactive map in browser' to view "
            f"the full interactive map with zoom and layers.")
        self._open_browser()

    def _open_browser(self):
        if self.html_path and os.path.exists(self.html_path):
            if sys.platform == "win32":
                os.startfile(self.html_path)
            else:
                subprocess.run(["xdg-open", self.html_path])

    def _open_folder(self):
        folder = "C:\\lsm_app\\outputs"
        if os.path.exists(folder):
            subprocess.run(["explorer", folder])