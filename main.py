import mediapipe as mp
import cv2
from pynput.keyboard import Controller as KeyboardController
from scipy.stats import linregress
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hands_area = [[], []]
counter = 0

leftHand_positions = [np.array([0, 0]), np.array([0, 0])]
rightHand_positions = [np.array([0, 0]), np.array([0, 0])]
damping_factory = 0.15

key = KeyboardController()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_list, y_list = [], []
            for finger_id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            top_left_point_hand = np.array([x_min, y_min])
            bottom_right_point_hand = np.array([x_max, y_max])

            handedness = results.multi_handedness[hand_id].classification[0].label
            handedness_id = -1
            hand_positions = []

            if handedness == "Left":
                current_top_left_point_hand = leftHand_positions[0] + (
                            top_left_point_hand - leftHand_positions[0]) * damping_factory
                current_bottom_right_point_hand = leftHand_positions[1] + (
                            bottom_right_point_hand - leftHand_positions[1]) * damping_factory

                handedness_id = 0
                hand_positions = leftHand_positions
            else:
                current_top_left_point_hand = rightHand_positions[0] + (
                            top_left_point_hand - rightHand_positions[0]) * damping_factory
                current_bottom_right_point_hand = rightHand_positions[1] + (
                            bottom_right_point_hand - rightHand_positions[1]) * damping_factory

                handedness_id = 1
                hand_positions = rightHand_positions

            max_width_hand = (current_bottom_right_point_hand[0] - current_top_left_point_hand[0])
            max_height_hand = (current_bottom_right_point_hand[1] - current_top_left_point_hand[1])
            hands_area[handedness_id] += [max_width_hand * max_height_hand]

            cv2.rectangle(
                img, (int(current_top_left_point_hand[0]), int(current_top_left_point_hand[1])),
                (int(current_bottom_right_point_hand[0]), int(current_bottom_right_point_hand[1])), (255, 255, 255),
                2
            )

            hand_positions[0] = current_top_left_point_hand
            hand_positions[1] = current_bottom_right_point_hand

    if len(hands_area[0]) >= 5 and len(hands_area[1]) >= 5:
        left = hands_area[0][-5:]
        right = hands_area[1][-5:]

        x = list(range(5))

        left_hand_slope, _, _, _, _ = linregress(x, left)
        right_hand_slope, _, _, _, _ = linregress(x, right)

        if left_hand_slope > 600:
            key.press('y')
            key.release('y')

            print(f"Left Click {counter}: {left_hand_slope}")
            counter += 1
        elif right_hand_slope > 600:
            key.press('u')
            key.release('u')

            print(f"Right Click {counter}: {right_hand_slope}")
            counter += 1

        hands_area[0] = []
        hands_area[1] = []

    cv2.imshow("Screen", img)

    pressed_key = cv2.waitKey(8)
    if pressed_key == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
