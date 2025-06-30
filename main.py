import cv2
import math
import mediapipe as mp
from math import hypot
import osascript
import numpy as np

cap = cv2.VideoCapture(0)  # Checks for camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:  # list of all hands detected.
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                # Get finger joint points
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        # getting the value at a point
        # x      #y
        x1, y1 = lmList[4][1], lmList[4][2]  # thumb
        x2, y2 = lmList[12][1], lmList[12][2]

        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)
        ang = 90 - math.degrees(math.asin((y2 - y1) / length))

        vol = np.interp(length, [30, 350], [0, 400]) * 3 / 4
        volbar = np.interp(length, [30, 350], [400, 150])
        volper = np.interp(length, [30, 350], [0, 100])


        print(vol, int(length))
        osascript.osascript("set volume output volume {}".format(vol))

        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

cap.release()  # stop cam
cv2.destroyAllWindows()  # close window
