import sys
import os

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    os.environ['GDAL_DATA'] = os.path.join(
        base_dir, 'gdal-data')
    os.environ['PROJ_LIB'] = os.path.join(
        base_dir, 'proj')
    os.environ['GDAL_DRIVER_PATH'] = os.path.join(
        base_dir, 'gdalplugins')
    os.environ['PROJ_DATA'] = os.path.join(
        base_dir, 'proj')

# ── Crash logger ───────────────────────────────────────────────
from logger import setup_crash_logger
crash_log = setup_crash_logger()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMessageBox)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from ui.upload_panel     import UploadPanel
from ui.fetch_panel      import FetchPanel
from ui.config_panel     import ConfigPanel
from ui.map_viewer       import MapViewer
from ui.results_panel    import ResultsPanel
from ui.comparison_panel import ComparisonPanel


def apply_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(30,  30,  30))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Base,            QColor(20,  20,  20))
    palette.setColor(QPalette.AlternateBase,   QColor(40,  40,  40))
    palette.setColor(QPalette.ToolTipBase,     QColor(220, 220, 220))
    palette.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
    palette.setColor(QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.Button,          QColor(45,  45,  45))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText,      QColor(255,  80,  80))
    palette.setColor(QPalette.Link,            QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight,       QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0,    0,   0))
    app.setPalette(palette)
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #444;
            border-radius: 6px;
            margin-top: 10px;
            font-weight: bold;
            color: #ccc;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QPushButton {
            background-color: #2a82da;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 5px 12px;
            font-weight: bold;
        }
        QPushButton:hover    { background-color: #3a92ea; }
        QPushButton:pressed  { background-color: #1a72ca; }
        QPushButton:disabled { background-color: #555; color: #888; }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 5px;
            text-align: center;
            color: white;
            background: #222;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            border-radius: 4px;
        }
        QTextEdit, QLineEdit {
            background-color: #1a1a1a;
            color: #ddd;
            border: 1px solid #444;
            border-radius: 4px;
        }
        QComboBox {
            background-color: #2d2d2d;
            color: #ddd;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 3px 8px;
        }
        QTabWidget::pane     { border: 1px solid #444; }
        QTabBar::tab {
            background: #2d2d2d;
            color: #aaa;
            padding: 8px 18px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { background: #1e1e1e; color: #fff; }
        QScrollArea          { border: none; }
        QSlider::groove:horizontal {
            height: 4px;
            background: #444;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #2a82da;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QLabel { color: #ccc; }
    """)


class LSMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "LSM Mapper — Landslide Susceptibility Mapping")
        self.setMinimumSize(1100, 750)

        self.upload_panel     = UploadPanel()
        self.fetch_panel      = FetchPanel()
        self.map_viewer       = MapViewer()
        self.results_panel    = ResultsPanel()
        self.comparison_panel = ComparisonPanel(
            self.upload_panel)
        self.config_panel     = ConfigPanel(
            self.upload_panel,
            self.map_viewer,
            self.results_panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.upload_panel,     "Upload data")
        self.tabs.addTab(self.fetch_panel,      "Auto-fetch")
        self.tabs.addTab(self.config_panel,     "Model settings")
        self.tabs.addTab(self.map_viewer,       "LSM Map")
        self.tabs.addTab(self.results_panel,    "Results & Charts")
        self.tabs.addTab(self.comparison_panel, "Model comparison")

        self.setCentralWidget(self.tabs)

        # Menu bar
        menubar   = self.menuBar()
        help_menu = menubar.addMenu("Help")

        from PyQt5.QtWidgets import QAction
        about_action = QAction("About LSM Mapper", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def switch_to_map(self):
        self.tabs.setCurrentIndex(3)

    def _show_about(self):
        from ui.about_dialog import AboutDialog
        dlg = AboutDialog(self)
        dlg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    try:
        from splash import create_splash
        splash = create_splash()
        splash.show()
        app.processEvents()

        import time
        time.sleep(2)

        window = LSMApp()
        splash.finish(window)
        window.show()

    except Exception as e:
        import traceback
        QMessageBox.critical(
            None,
            "Startup Error",
            f"LSM Mapper failed to start:\n\n"
            f"{traceback.format_exc()}\n\n"
            f"Check crash_log.txt for details.")
        sys.exit(1)

    sys.exit(app.exec_())