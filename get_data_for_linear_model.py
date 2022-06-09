import cv2
import numpy as np
import glob
import cv2.aruco as aruco
import sys, time, math

#define tag
id_to_fine = 23
marker_size = 15

# get camera calibration path
camera_matrix = np.loadtxt('cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt('cameraDistortion.txt', delimiter=',')

# define aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find all the aruco maker in the image
    corners, ids, rejected = aruco.detectMarkers(image= gray, dictionary=aruco_dict, parameters = parameters,
                                                cameraMatrix = camera_matrix, distCoeff = camera_distortion)
    if len(corners) != 0  and ids[0] == id_to_fine:
        #-- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        # draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(img,corners)
        aruco.drawAxis(img, camera_matrix, camera_distortion, rvec, tvec, 10)

        # print the tag position in camera frame
        str_position = "Marker position x=%4.0f y= %4.0f z=%4.0f"%(tvec[0],tvec[1],tvec[2])
        cv2.putText(img, str_position,(0,100),font, 1, (0,250,0),2, cv2.LINE_AA)


    # display the frame
    cv2.line(img,(320,240),(390,240),(0,0,255),5)
    cv2.line(img,(320,240),(320,310),(0,255,0),5)

    cv2.imshow('frame', img)

    # use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

