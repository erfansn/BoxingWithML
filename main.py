import mediapipe as mp
import cv2
from pynput.keyboard import Controller as KeyboardController
from scipy.stats import linregress

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hand_area = [[], []]
hand_order = [0, 0]
counter = 0

position_leftHand = (0, 0)
position_rightHand = (0, 0)

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

            # cv2.rectangle(
            #     img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2
            # )

            if hand_id < 2:
                hand_area[hand_id] += [(x_max - x_min) * (y_max - y_min)]
                hand_order[hand_id] = x_min

    if len(hand_area[0]) and len(hand_area[1]) >= 5:
        left = hand_area[0][-5:]
        right = hand_area[1][-5:]

        if hand_order[0] > hand_order[1]:
            left, right = right, left

        x = list(range(5))

        left_hand_slope, _, _, _, _ = linregress(x, left)
        right_hand_slope, _, _, _, _ = linregress(x, right)

        # print(f"Left: {left_hand_slope}, Right: {right_hand_slope}")
        # print(left_hand_slope)

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

        hand_area[0] = []
        hand_area[1] = []

    cv2.imshow("Screen", img)

    pressed_key = cv2.waitKey(8)
    if pressed_key == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
