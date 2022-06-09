import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(img, markerSize = 6, totalMarkers = 250, draw = True):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bbox, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        aruco.drawDetectedMarkers(img, bbox)
    return [bbox, ids]
# def argument
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        arucoFound = findArucoMarkers(img)
        print(len(arucoFound[0]))

        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                print("hello")
        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

