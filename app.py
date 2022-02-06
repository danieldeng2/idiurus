from PySide6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QBrush
from PySide6.QtCore import Signal, Slot, Qt, QThread, QObject
from pynput.mouse import Button, Controller
from screeninfo import get_monitors
from sys import platform
from numpy import average
import sys
import cv2
import math
import numpy as np
import mediapipe as mp

# Mediapipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Screen size calculation
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

DEVICE_INDEX = -1 if platform == "linux" else 0

mouse = Controller()

class VideoThread(QThread):
    finished = Signal()
    change_pixmap_signal = Signal(np.ndarray)

    leftClick = False
    rightClick = False
    scrolling = False
    taking_screenshot = False
    scrollingState = (0,0)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # Capture feed from webcam
        cap = cv2.VideoCapture(DEVICE_INDEX)

        pointer_xs = []
        pointer_ys = []

        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self._run_flag:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                    # Extra fingers (not used)
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    index_x = 1 - index_tip.x
                    thumb_x = 1 - thumb_tip.x

                    index_y = index_tip.y
                    thumb_y = thumb_tip.y

                    pointer_xs.append((index_x + thumb_x) / 2)
                    pointer_ys.append((index_y + thumb_y) / 2)

                    if len(pointer_xs) > 5:
                        pointer_xs.pop(0)
                        pointer_ys.pop(0)

                    pointer_x = average(pointer_xs)
                    pointer_y = average(pointer_ys)
                    
                    crop_ratio = 0.2

                    pointer_x = (pointer_x - crop_ratio) / (1 - 2 * crop_ratio)
                    pointer_y = (pointer_y - crop_ratio) / (1 - 2 * crop_ratio)

                    pointer_x = pointer_x * SCREEN_WIDTH
                    pointer_y = pointer_y * SCREEN_HEIGHT

                    index_thumb_distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

                    if index_thumb_distance < 0.1 and not self.leftClick:
                        mouse.press(Button.left)
                    if index_thumb_distance > 0.1 and self.leftClick:
                        mouse.release(Button.left)
                    self.leftClick = index_thumb_distance < 0.1

                    mouse.position = (pointer_x, pointer_y)
                self.change_pixmap_signal.emit(cv2.flip(image, 1))

        # Shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.finished.emit()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magic Wand")
        self.disply_width = 640
        self.display_height = 480
        
        # Placeholder
        self.placeholder = self.convert_cv_qt(cv2.imread('placeholder.png'))  

        # Create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.roundedCorners(self.placeholder))
        self.image_label.resize(self.disply_width, self.display_height)
        
        # Create a text label
        self.text_label = QLabel('Webcam')

        # Button to pause and resume video
        self.capture_button = QPushButton(self)
        self.capture_button.setText('Start Capture')
        self.capture_button.resize(300, 100)
        self.capture_button.clicked.connect(self.captureToggle)
        self.capturing = False

        # Create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.capture_button)
        
        # Set the vbox layout as the widgets layout
        self.setLayout(vbox)   

    def roundedCorners(self, pixmap):
        rounded = QPixmap(pixmap.size())
        rounded.fill(QColor("transparent"))

        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(pixmap))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(pixmap.rect(), 8, 8)

        return rounded

    def captureToggle(self):
        if self.capturing:
            self.thread.stop()
            self.capture_button.setText('Resume Capture')
            self.capturing = False
            return

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.finished.connect(self.paused)
        self.thread.start()
        self.capture_button.setText('Pause Capture')
        self.capturing = True

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(self.roundedCorners(qt_img))
    
    def paused(self):
        self.image_label.setPixmap(self.roundedCorners(self.placeholder))
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())