import sys
import time
from pathlib import Path

import mss
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QFont, QColor

# Détermination de la zone de capture (écran principal)
with mss.mss() as sct:
    # Sélection du premier moniteur (index 1) qui correspond généralement à l'écran principal
    monitor = sct.monitors[1]

class OverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Fenêtre sans bordure, toujours au-dessus, n’apparaissant pas dans la barre des tâches
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        # Fond de fenêtre transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Laisser passer les événements de souris à travers la fenêtre (overlay non bloquant)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        # Ajuster la fenêtre à la taille de l'écran principal
        self.setGeometry(monitor["left"], monitor["top"], monitor["width"], monitor["height"])
        self.setFixedSize(monitor["width"], monitor["height"])
        # Variables pour stocker les résultats de détection à afficher
        self.boxes = []         # Liste des boîtes [x_min, y_min, x_max, y_max]
        self.class_names = []   # Liste des noms de classe pour chaque boîte
        self.confidences = []   # Liste des scores de confiance pour chaque boîte

    def update_boxes(self, boxes, class_names=None, confidences=None):
        """Met à jour les données de détection et rafraîchit l'affichage."""
        self.boxes = boxes
        self.class_names = class_names if class_names is not None else []
        self.confidences = confidences if confidences is not None else []
        self.update()  # Déclenche paintEvent pour redessiner l'overlay

    def paintEvent(self, event):
        # Dessin des boîtes de détection en rouge
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(Qt.red)
        pen.setWidth(3)  # épaisseur du contour
        painter.setPen(pen)
        for box in self.boxes:
            x, y, x2, y2 = box
            width = x2 - x
            height = y2 - y
            painter.drawRect(x, y, width, height)
        painter.end()

        # Dessin des étiquettes de classe et scores au-dessus de chaque boîte
        if not self.boxes or not self.class_names or not self.confidences:
            return  # rien à dessiner si aucune détection

        painter_text = QPainter(self)
        painter_text.setRenderHint(QPainter.Antialiasing)
        # Style du texte (fond semi-transparent noir, texte blanc en Arial gras 10)
        painter_text.setBrush(QColor(0, 0, 0, 150))
        painter_text.setPen(Qt.NoPen)
        font = QFont("Arial", 10)
        font.setBold(True)
        painter_text.setFont(font)
        margin = 4  # marge autour du texte

        for (x, y, x2, y2), cls_name, conf in zip(self.boxes, self.class_names, self.confidences):
            # Préparation du texte "classe: score"
            label = f"{cls_name}: {conf:.2f}"
            # Calcul des dimensions du rectangle de fond du texte
            metrics = painter_text.fontMetrics()
            text_width = metrics.boundingRect(label).width()
            text_height = metrics.height()
            # Position du texte (au-dessus de la boîte, aligné à gauche)
            text_x = x
            text_y = y - text_height - margin
            if text_y < 0:
                text_y = y + margin  # si trop en haut, on le place en dessous de la boîte
            # Dessiner le fond rectangulaire semi-transparent
            painter_text.drawRect(text_x, text_y, text_width + 2*margin, text_height + margin)
            # Dessiner le texte en blanc
            painter_text.setPen(Qt.white)
            painter_text.drawText(text_x + margin, text_y + text_height - metrics.descent(), label)
            painter_text.setPen(Qt.NoPen)  # réinitialiser le stylo pour le prochain texte
        painter_text.end()

class DetectionThread(QThread):
    """Thread gérant la capture d'écran et la détection YOLO en continu."""
    update_signal = pyqtSignal(list, list, list)  # signal émettant (boxes, class_names, confidences)

    def __init__(self, model, monitor_region):
        super().__init__()
        self.model = model
        self.monitor = monitor_region
        self.running = True

    def run(self):
        with mss.mss() as sct:
            while self.running:
                # 1. Capture de l'écran (zone définie par monitor)
                screenshot = sct.grab(self.monitor)
                frame = np.array(screenshot)  # image en numpy (avec canaux BGRA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # convertir en BGR pour OpenCV/YOLO
                # 2. Inférence YOLO sur l'image capturée
                results = self.model.predict(source=frame, verbose=False)  # pas d'affichage, mode silencieux
                result = results[0]  # on traite un seul résultat (image unique)
                # 3. Traitement des résultats de détection
                if result.boxes.cls.shape[0] == 0:
                    # Aucune détection trouvée
                    self.update_signal.emit([], [], [])
                else:
                    # Extraction des coordonnées des boîtes (converties en int)
                    boxes = np.array(result.boxes.xyxy.cpu(), dtype=int).tolist()
                    # Extraction des noms de classe et des scores de confiance
                    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
                    class_names = [self.model.names[c] for c in class_ids]
                    confidences = result.boxes.conf.cpu().numpy().tolist()
                    # Émission des résultats pour mise à jour de l'overlay
                    self.update_signal.emit(boxes, class_names, confidences)
                # 4. Petite pause pour réduire la charge et assurer la fluidité (à ajuster si besoin)
                time.sleep(0.1)

    def stop(self):
        """Demande l'arrêt du thread proprement."""
        self.running = False
        self.wait()  # attend la fin de run() si en cours

if __name__ == "__main__":
    # Initialisation de l'application PyQt5
    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()

    # Chargement du modèle YOLO entraîné : best.pt à côté du script, ou le chemin passé en argument
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "best.pt"
    model = YOLO(str(model_path))  # charge le modèle personnalisé

    # Démarrage du thread de détection en temps réel
    detection_thread = DetectionThread(model, monitor)
    detection_thread.update_signal.connect(overlay.update_boxes)  # connexion du signal aux mises à jour de l'overlay
    detection_thread.start()

    # Assurer l'arrêt correct du thread lors de la fermeture de l'application
    app.aboutToQuit.connect(detection_thread.stop)
    sys.exit(app.exec_())
