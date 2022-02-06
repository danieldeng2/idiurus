import cv2
import mediapipe as mp
import math
import enum
from sys import platform
from numpy import average
from pynput.mouse import Button, Controller
from pynput.keyboard import Key
from pynput.keyboard import Controller as KeyboardController
from screeninfo import get_monitors
from datetime import datetime
import pyautogui

keyboard = KeyboardController()

pointer_xs = []
pointer_ys = []

index_x_history = [-1, -1, -1, -1, -1]
last_media_change = datetime.now()

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

def distance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Action(enum.Enum):
    resting = 1
    leftclick = 2
    scroll = 3
currentState = Action.resting

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

                if index_thumb_distance < 0.1:
                    clickState = (mcp_x, mcp_y)
                    mouse.press(Button.left)
                    currentState = Action.leftclick
                else:
                    if currentState == currentState.leftclick:
                        mouse.release(Button.left)
                        currentState = Action.resting

                # if currentState == currentState.leftclick:
                    # print(distance(mcp_x, mcp_y, clickState[0], clickState[1]))
                    # if distance(mcp_x, mcp_y, clickState[0], clickState[1]) < 0.05:
                    #     movePointer = False

                # leftMousePressed = index_thumb_distance < 0.1
                # if ring_thumb_distance > 0.1 and rightMousePressed
                #
                #                 # if ring_thumb_distance < 0.1 and not rightMousePressed:
                #                 #     mouse.press(Button.right):
                #     mouse.release(Button.right)
                # rightMousePressed = ring_thumb_distance < 0.1

                # if thumb_middle_distance < 0.2 and not scrolling:
                #     scrollingState = (thumb_x, thumb_y)
                # scrolling = thumb_middle_distance < 0.2
                if thumb_middle_distance < 0.2:
                    if currentState == Action.resting:
                        currentState = Action.scroll
                        scrollingState = (mcp_x, mcp_y)
                else:
                    if currentState == Action.scroll:
                        currentState = Action.resting

                if currentState == Action.scroll:
                    # print("CURRENTLY SCROLLING")
                    # print(thumb_y)
                    # print(scrollingState[1])
                    # print("DISTANCE: ", thumb_y - scrollingState[1])
                    if thumb_y - scrollingState[1] > 0.2:
                        # scroll up
                        mouse.scroll(0, -1)
                    elif scrollingState[1] - thumb_y > 0.2:
                        # scroll down
                        mouse.scroll(0, 1)
                    movePointer = False
                
                if movePointer:
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
