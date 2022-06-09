import cv2
import numpy as np
import glob
import cv2.aruco as aruco
import sys, time, math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

#define tag
id_to_fine = 23
marker_size = 10

# get camera calibration path
camera_matrix = np.loadtxt('cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt('cameraDistortion.txt', delimiter=',')

# 180 deg rotation matrix around the x axis
R_flip = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] =  1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0


# define aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

# ..... capture the video
cap = cv2.VideoCapture(0)
# set camera size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# def draw_cub(img, corners, imgpts):
#     imgpts = np.int32(imgpts).reshape(-1,2)
#     # draw ground floor in green
#     img = cv2.drawContours(img,[imgpts[:4]],-1,(0,255,0),-3)
#     # draw pillars in blue color
#     for i,j in zip(range(4), range(4,8)):
#         img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
font = cv2.FONT_HERSHEY_PLAIN
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

        # obtain the rotation mat atg -> camera
        Rct = np.matrix(cv2.Rodrigues(rvec)[0])
        Rtc = Rct.T

        # get the attitude
        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*Rtc)

        # print the marker'attitude respect to camera frame
        strAttitude = "Marker Attitude r=%4.0f p=%4.0f y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                                                                 math.degrees(yaw_marker))
        cv2.putText(img, strAttitude,(0,300),font, 1, (0,250,0),2, cv2.LINE_AA)


    # display the frame
    # cv2.line(img,(320,240),(390,240),(0,0,255),5)
    # cv2.line(img,(320,240),(320,310),(0,255,0),5)

    cv2.imshow('frame', img)

    # use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
