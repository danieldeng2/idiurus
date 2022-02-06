from PySide6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QBrush
from PySide6.QtCore import Signal, Slot, Qt, QThread, QObject
from pynput.mouse import Button, Controller
from pynput.keyboard import Key
from pynput.keyboard import Controller as KeyboardController
from screeninfo import get_monitors
from datetime import datetime
from sys import platform
from numpy import average
import sys
import cv2
import math
import numpy as np
import mediapipe as mp
import enum

keyboard = KeyboardController()

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

class Action(enum.Enum):
    resting = 1
    leftclick = 2
    scroll = 3

class VideoThread(QThread):
    finished = Signal()
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        pointer_xs = []
        pointer_ys = []

        def weighted_input(inputs, input):
            inputs.append(input)
            if (len(inputs) > 10):
                inputs.pop(0)

            weights = 0
            total = 0
            for i, input in enumerate(inputs):
                weight = 1 / (len(inputs) - i)
                total += input * weight
                weights += weight

            return total / weights
        
        def distance(x1, x2, y1, y2):
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


        # Capture feed from webcam
        index_x_history = [-1, -1, -1, -1, -1]
        screen_x = monitor.width
        screen_y = monitor.height
        currentState = Action.resting
        last_media_change = datetime.now()

        cap = cv2.VideoCapture(DEVICE_INDEX)

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
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                        index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                        index_x = 1 - index_tip.x
                        thumb_x = 1 - thumb_tip.x
                        middle_x = 1 - middle_finger_tip.x
                        ring_x = 1 - ring_finger_tip.x
                        mcp_x = 1 - index_finger_mcp.x
                        pinky_x = 1 - pinky_tip.x

                        index_y = index_tip.y
                        thumb_y = thumb_tip.y
                        middle_y = middle_finger_tip.y
                        mcp_y = index_finger_mcp.y
                        ring_y = ring_finger_tip.y
                        pinky_y = pinky_tip.y

                        pointer_x = weighted_input(pointer_xs, mcp_x)
                        pointer_y = weighted_input(pointer_ys, mcp_y)

                        crop_ratio = 0.2

                        pointer_x = (pointer_x - crop_ratio) / (1 - 2 * crop_ratio)
                        pointer_y = (pointer_y - crop_ratio) / (1 - 2 * crop_ratio)

                        pointer_x = pointer_x * screen_x
                        pointer_y = pointer_y * screen_y

                        index_thumb_distance = distance(index_x, thumb_x, index_y, thumb_y)
                        thumb_middle_distance = distance(middle_x, thumb_x, middle_y, thumb_y)
                        ring_thumb_distance = distance(ring_x, thumb_x, ring_y, thumb_y)
                        pinky_thumb_distance = distance(pinky_x, thumb_x, pinky_y, thumb_y)

                        movePointer = True

                        # Record the previous index tip.
                        previous_x = index_x_history.pop(0)
                        index_x_history.append(index_x)
                        if (datetime.now() - last_media_change).total_seconds() > 2:
                            if previous_x != -1 and index_x - previous_x > 0.15 and last_media_change:
                                print("*********RIGHT!")
                                keyboard.press(Key.media_next)
                                last_media_change = datetime.now()
                            elif previous_x != -1 and index_x - previous_x < -0.15:
                                print("*********LEFT!")
                                keyboard.press(Key.media_previous)
                                last_media_change = datetime.now()
                            # else:
                            #     print("SLOW!")

                        # print(pinky_thumb_distance)
                        if pinky_thumb_distance < 0.1:
                            if currentState == Action.resting:
                                currentState = Action.scroll
                                scrollingState = (mcp_x, mcp_y)
                        else:
                            if currentState == Action.scroll:
                                currentState = Action.resting

                        if index_thumb_distance < 0.1:
                            if currentState == Action.resting:
                                clickState = (mcp_x, mcp_y)
                                mouse.press(Button.left)
                                currentState = Action.leftclick
                        else:
                            if currentState == Action.leftclick:
                                mouse.release(Button.left)
                                currentState = Action.resting

                        if currentState == currentState.leftclick:
                            movePointer = thumb_middle_distance < 0.2

                        if currentState == Action.scroll:
                            if thumb_y - scrollingState[1] > 0.1:
                                # scroll down
                                mouse.scroll(0, 1)
                            elif scrollingState[1] - thumb_y > 0.1:
                                # scroll up
                                mouse.scroll(0, -1)
                            movePointer = False
                        
                        if movePointer:
                            mouse.position = (pointer_x, pointer_y)

                        # mp_drawing.draw_landmarks(
                        #     image,
                        #     hand_landmarks,
                        #     mp_hands.HAND_CONNECTIONS,
                        #     mp_drawing_styles.get_default_hand_landmarks_style(),
                        #     mp_drawing_styles.get_default_hand_connections_style(),
                        # )
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