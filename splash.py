from PyQt5.QtWidgets import QSplashScreen, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont

def create_splash():
    pixmap = QPixmap(600, 350)
    pixmap.fill(QColor(20, 20, 20))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Top accent bar
    painter.fillRect(0, 0, 600, 6, QColor(42, 130, 218))

    # App title
    painter.setPen(QColor(255, 255, 255))
    font = QFont("Segoe UI", 28, QFont.Bold)
    painter.setFont(font)
    painter.drawText(0, 60, 600, 60,
                     Qt.AlignCenter, "LSM Mapper")

    # Subtitle
    painter.setPen(QColor(170, 170, 170))
    font2 = QFont("Segoe UI", 12)
    painter.setFont(font2)
    painter.drawText(0, 120, 600, 40,
                     Qt.AlignCenter,
                     "Landslide Susceptibility Mapping Tool")

    # Divider line
    painter.setPen(QColor(60, 60, 60))
    painter.drawLine(60, 175, 540, 175)

    # Info lines
    painter.setPen(QColor(120, 120, 120))
    font3 = QFont("Segoe UI", 9)
    painter.setFont(font3)
    painter.drawText(0, 190, 600, 30,
                     Qt.AlignCenter,
                     "Machine Learning based GIS Desktop Application")
    painter.drawText(0, 215, 600, 30,
                     Qt.AlignCenter,
                     "Supports Random Forest  |  XGBoost  |  Logistic Regression")

    # Version
    painter.setPen(QColor(42, 130, 218))
    font4 = QFont("Segoe UI", 9, QFont.Bold)
    painter.setFont(font4)
    painter.drawText(0, 255, 600, 30,
                     Qt.AlignCenter, "Version 1.0")

    # Loading text
    painter.setPen(QColor(90, 90, 90))
    font5 = QFont("Segoe UI", 8)
    painter.setFont(font5)
    painter.drawText(0, 310, 600, 30,
                     Qt.AlignCenter, "Loading...")

    # Bottom accent bar
    painter.fillRect(0, 344, 600, 6, QColor(42, 130, 218))

    painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setMask(pixmap.mask())
    return splash