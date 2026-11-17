import sys
from pathlib import Path

import numpy as np
import cv2
import mss
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import YOLO
import torch

# Poids par défaut : best.pt à côté du script. Un autre chemin peut être passé en argument.
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "best.pt"

# YOLOv11 (CUDA si dispo)
model = YOLO(str(MODEL_PATH))
if torch.cuda.is_available():
    model.to("cuda")


class YoloThread(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.running = True
        self.monitor = mss.mss().monitors[1]
        self.scale_x = 1
        self.scale_y = 1

    def run(self):
        with mss.mss() as sct:
            while self.running:
                screenshot = np.array(sct.grab(self.monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                self.scale_x = frame.shape[1] / small_frame.shape[1]
                self.scale_y = frame.shape[0] / small_frame.shape[0]

                results = model(small_frame, conf=0.75)

                boxes = []
                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = box
                    x1 = int(x1 * self.scale_x)
                    y1 = int(y1 * self.scale_y)
                    x2 = int(x2 * self.scale_x)
                    y2 = int(y2 * self.scale_y)
                    boxes.append((x1, y1, x2, y2))

                self.result_ready.emit(boxes)

    def stop(self):
        self.running = False
        self.wait()


class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        self.screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(self.screen)

        self.boxes = []
        self.previous_boxes = []
        self.alpha = 0.5  # Lissage : plus proche de 1 = plus lent mais plus stable

        self.worker = YoloThread()
        self.worker.result_ready.connect(self.update_boxes)
        self.worker.start()

    def update_boxes(self, new_boxes):
        smoothed_boxes = []

        if not self.previous_boxes or len(new_boxes) != len(self.previous_boxes):
            smoothed_boxes = new_boxes
        else:
            for new, old in zip(new_boxes, self.previous_boxes):
                x1 = int(self.alpha * old[0] + (1 - self.alpha) * new[0])
                y1 = int(self.alpha * old[1] + (1 - self.alpha) * new[1])
                x2 = int(self.alpha * old[2] + (1 - self.alpha) * new[2])
                y2 = int(self.alpha * old[3] + (1 - self.alpha) * new[3])
                smoothed_boxes.append((x1, y1, x2, y2))

        self.previous_boxes = smoothed_boxes
        self.boxes = smoothed_boxes
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtGui.QColor(0, 200, 0, 120))
        pen.setWidth(2)
        painter.setPen(pen)

        for (x1, y1, x2, y2) in self.boxes:
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
