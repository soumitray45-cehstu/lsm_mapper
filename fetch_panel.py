from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QGroupBox
)

class FetchPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter bounding box coordinates to auto-fetch data from Google Earth Engine:"))

        coords_group = QGroupBox("Bounding box (decimal degrees)")
        coords_layout = QHBoxLayout()

        self.min_lon = QLineEdit(); self.min_lon.setPlaceholderText("Min Lon (e.g. 91.5)")
        self.min_lat = QLineEdit(); self.min_lat.setPlaceholderText("Min Lat (e.g. 22.0)")
        self.max_lon = QLineEdit(); self.max_lon.setPlaceholderText("Max Lon (e.g. 92.5)")
        self.max_lat = QLineEdit(); self.max_lat.setPlaceholderText("Max Lat (e.g. 23.0)")

        for w in [self.min_lon, self.min_lat, self.max_lon, self.max_lat]:
            coords_layout.addWidget(w)
        coords_group.setLayout(coords_layout)
        layout.addWidget(coords_group)

        date_group = QGroupBox("Date range")
        date_layout = QHBoxLayout()
        self.start_date = QLineEdit(); self.start_date.setPlaceholderText("Start (YYYY-MM-DD)")
        self.end_date   = QLineEdit(); self.end_date.setPlaceholderText("End   (YYYY-MM-DD)")
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(self.end_date)
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)

        fetch_btn = QPushButton("Fetch data from GEE")
        fetch_btn.setFixedHeight(40)
        fetch_btn.clicked.connect(self._fetch)
        layout.addWidget(fetch_btn)

        self.status = QLabel("")
        layout.addWidget(self.status)
        layout.addStretch()
        self.setLayout(layout)

    def _fetch(self):
        self.status.setText("Connecting to Google Earth Engine... (make sure ee.Authenticate() was run once)")