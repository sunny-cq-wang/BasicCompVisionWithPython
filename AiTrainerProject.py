import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

prevTime = 0

count = 0
dir = 0


while True:
    success, img = cap.read()

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        # detector.findAngle(img, 12, 14, 16)  # right arm
        angle = detector.findAngle(img, 11, 13, 15)  # left arm
        per = np.interp(angle, (170, 280), (0, 100))
        bar = np.interp(angle, (170, 280), (350, 30))
        # print(per, angle)

        # check for dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        # draw bar
        cv2.rectangle(img, (600, 30), (630, 350), color, 3)
        cv2.rectangle(img, (600, int(bar)), (630, 350), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (595, 17), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        # draw curl count
        cv2.rectangle(img, (0, 300), (150, 480), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (30, 420), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)



    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
