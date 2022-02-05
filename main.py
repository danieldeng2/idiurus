import cv2
import mediapipe as mp
import math

from sys import platform
from numpy import average
from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import pyautogui

pointer_xs = []
pointer_ys = []
def weighted_input(inputs, input):
    inputs.append(input)
    if (len(inputs) > 15):
        inputs.pop(0)

    weights = 0
    total = 0
    for i, input in enumerate(inputs):
        weight = 1 / (len(inputs) - i)
        total += input * weight
        weights += weight

    return total / weights

# Screen size calculation
monitor = get_monitors()[0]
screen_x = monitor.width
screen_y = monitor.height

mouse = Controller()
leftMousePressed = False
rightMousePressed = False
scrolling = False
taking_screenshot = False
scrollingState = (0,0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print("Handedness:", results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            mouse.move(finger_tip.x, finger_tip.y)
            print("hand_landmarks:", hand_landmarks)
            print(
                f"Index finger tip coordinates: (", f"{finger_tip.x * image_width}, " f"{finger_tip.y * image_height})"
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        cv2.imwrite("/tmp/annotated_image" + str(idx) + ".png", cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:

device_index = -1 if platform == "linux" else 0
cap = cv2.VideoCapture(device_index)


with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            # Uncomment to enable screenshot
            # if len(results.multi_hand_landmarks) > 1:
            #   first_hand_landmarks = results.multi_hand_landmarks[0]
            #   second_hand_landmarks = results.multi_hand_landmarks[1]
            #   first_finger_tip = first_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            #   second_finger_tip = second_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            #   two_index_distance = math.sqrt((first_finger_tip.x - second_finger_tip.x)**2 + (first_finger_tip.y - second_finger_tip.y)**2)
            #   if two_index_distance < 0.05:
            #     if not taking_screenshot:
            #       taking_screenshot = True
            #       myScreenshot = pyautogui.screenshot()
            #       myScreenshot.save(r'screenshot.png')
            #   else:
            #     taking_screenshot = False
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                
                
                index_x = 1 - index_tip.x
                thumb_x = 1 - thumb_tip.x
                middle_x = 1 - middle_finger_tip.x
                ring_x = 1 - ring_finger_tip.x

                index_y = index_tip.y
                thumb_y = thumb_tip.y
                middle_y = middle_finger_tip.y
                ring_y = 1 - ring_finger_tip.y

                pointer_x = weighted_input(pointer_xs, (index_x + thumb_x) / 2)
                pointer_y = weighted_input(pointer_ys, (index_y + thumb_y) / 2)
                
                crop_ratio = 0.2

                pointer_x = (pointer_x - crop_ratio) / (1 - 2 * crop_ratio)
                pointer_y = (pointer_y - crop_ratio) / (1 - 2 * crop_ratio)

                pointer_x = pointer_x * screen_x
                pointer_y = pointer_y * screen_y

                index_thumb_distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
                thumb_middle_distance = math.sqrt((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2)
                ring_thumb_distance = math.sqrt((ring_x - thumb_x) ** 2 + (ring_y - thumb_y) ** 2)

                if index_thumb_distance < 0.1 and not leftMousePressed:
                    mouse.press(Button.left)
                if index_thumb_distance > 0.1 and leftMousePressed:
                    mouse.release(Button.left)
                leftMousePressed = index_thumb_distance < 0.1

                # if ring_thumb_distance < 0.1 and not rightMousePressed:
                #     mouse.press(Button.right)
                # if ring_thumb_distance > 0.1 and rightMousePressed:
                #     mouse.release(Button.right)
                # rightMousePressed = ring_thumb_distance < 0.1

                if thumb_middle_distance < 0.2 and not scrolling:
                    scrollingState = (thumb_x, thumb_y)
                scrolling = thumb_middle_distance < 0.2

                if scrolling:
                    if thumb_y - scrollingState[1] > 0.2:
                        # scroll up
                        mouse.scroll(0, -1)
                    elif scrollingState[1] - thumb_y > 0.2:
                        # scroll down
                        mouse.scroll(0, 1)
                else:
                    mouse.position = (pointer_x, pointer_y)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
