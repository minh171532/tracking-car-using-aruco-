import cv2
import time
import sys
import argparse
import os

def save_snap(width=0, height=0, name="snapshot", folder = "save_images"):
    cap = cv2.VideoCapture(0)
    if width > 0 and height > 0:
        print("setting the custom width and height")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    nSnap = 0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fileName = "%s/%s_%d_%d_"%(folder, name, w,h)

    while True:
        ret, frame = cap.read()
        cv2.imshow('camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            print("saving image ", str(nSnap))
            cv2.imwrite("%s%d.jpg"%(fileName,nSnap), frame)
            nSnap += 1
    cap.release()
    cv2.destroyAllWindows()



save_snap(width=1920, height=1080)







