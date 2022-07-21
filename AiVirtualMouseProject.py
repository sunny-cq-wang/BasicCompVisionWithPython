import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

#############################################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#############################################

prevTime = 0
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(maxHands=1)

wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get type of index & middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        X2, Y2 = lmList[12][1:]

    # 3. Check which fingers are up
    fingers = detector.fingersUp()

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    # 4. Only index finger up : moving mode
    if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

        # 6. Smooth Values
        currLocX = prevLocX + (x3 - prevLocX) / smoothening
        currLocY = prevLocY + (y3 - prevLocY) / smoothening

        #7. Move Mouse
        autopy.mouse.move(wScr-currLocX, currLocY)  # flipping coordinates so moving hand left moves mouse left too, will be reversed on camera screen
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        prevLocX, prevLocY = currLocX, currLocY


    # 8. Both index & middle fingers up: clicking mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(12, 8, img, r=10)
        print(length)
        # 10. Click mouse if distance
        if length < 30:
            cv2.circle(img, (lineInfo[-2], lineInfo[-1]), 10, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()


    # 11. Frame Rate
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

