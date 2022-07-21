import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

####################################
brushThickness = 15
eraserThickness = 50
####################################

# importing images
folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[0]  # default image
drawColor = (255, 0, 255)

IMG_WIDTH = 1280
IMG_HEIGHT = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

detector = htm.HandDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        x1, y1 = lmList[8][1:]  # tip of index finger
        x2, y2 = lmList[12][1:]  # tip of middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection mode (2 finger up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print('Selection Mode')
            if y1 < 120:
                # fingers in header, change color / eraser
                if 200 < x1 < 250:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)  # purple
                elif 375 < x1 < 425:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)  # blue
                elif 550 < x1 < 600:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)  # green
                elif 750 < x1 < 825:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # black for eraser

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode (index finger up)
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)



    # setting up the header iamge
    h, w, c = header.shape
    img[0:h, 0: w] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)