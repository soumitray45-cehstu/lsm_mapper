from PyQt5.QtGui import QPalette, QColor


def apply_theme(app):
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
