import cv2
import numpy as np
import glob
import cv2.aruco as aruco
import sys, time, math
from utilis import*

##########################################################
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

def get_center(corners):
    # corner.shape : (1,4,2)
    x = int((corners[0][0][0] + corners[0][2][0])/2)
    y = int((corners[0][0][1] + corners[0][2][1])/2)
    return (x,y)


###########################################################
# define tag
# id_to_fine = 25
marker_size = 15

# get camera calibration path
camera_matrix = np.loadtxt('cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt('cameraDistortion.txt', delimiter=',')

# 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# define aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

# ..... capture the video
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
flag = True
if flag:
    brightness, contrast, saturation, gain, exposure, white_balance, focus = np.loadtxt('camera_parameter.txt', delimiter=',').astype(int)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 15)
    # set camera size

    cap.set(10, brightness) # brightness     min: 0   , max: 255 , increment:1
    cap.set(11, contrast) # contrast       min: 0   , max: 255 , increment:1
    cap.set(12, saturation) # saturation     min: 0   , max: 255 , increment:1
    # cap.set(13, 13) # hue
    cap.set(14, gain) # gain           min: 0   , max: 127 , increment:1
    cap.set(15, exposure) # exposure       min: -7  , max: -1  , increment:1
    cap.set(17, white_balance) # white_balance  min: 4000, max: 7000, increment:1
    cap.set(28, focus) # focus          min: 0   , max: 255 , increment:5
font = cv2.FONT_HERSHEY_PLAIN

while True:

    ret, img = cap.read()
    # img = cv2.imread("test_img.jpg")
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find all the aruco maker in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                 cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    ctr_point1 = None
    ctr_point2 = None
    ctr_point3 = None 
    ctr_point4 = None

    if len(corners) != 0:
        for i in range(0, len(ids)):
            if ids[i]== 25:
                corner1 = corners[i]
                ctr_point1 = get_center(corner1)
                # -- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera frame
                rvec1, tvec1, markerPoints1 = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix,
                                                                           camera_distortion)
                # draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, camera_matrix, camera_distortion, rvec1, tvec1, 10)

                # print the tag position in camera frame
                str_position = "Marker position x=%4.0f y= %4.0f z=%4.0f"%(tvec1[0][0][0],tvec1[0][0][1],tvec1[0][0][2])
                cv2.putText(img, str_position,(0,100),font, 1, (0,250,0),4, cv2.LINE_AA)


            if ids[i]==35:
                corner2 = corners[i]
                ctr_point2 = get_center(corner2)

                # -- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
                rvec2, tvec2, markerPoints2 = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix,
                                                                           camera_distortion)
                # draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, camera_matrix, camera_distortion, rvec2, tvec2, 10)

                # print the tag position in camera frame
                str_position = "Marker position x=%4.0f y= %4.0f z=%4.0f"%(tvec2[0][0][0],tvec2[0][0][1],tvec2[0][0][2])
                cv2.putText(img, str_position,(0,300),font, 1, (0,250,0),4, cv2.LINE_AA)

            if ids[i]==45:
                corner3 = corners[i]
                ctr_point3 = get_center(corner3)

                # -- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
                rvec3, tvec3, markerPoints3 = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix,
                                                                           camera_distortion)
                # draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, camera_matrix, camera_distortion, rvec3, tvec3, 10)

                # print the tag position in camera frame
                str_position = "Marker position x=%4.0f y= %4.0f z=%4.0f"%(tvec3[0][0][0],tvec3[0][0][1],tvec3[0][0][2])
                cv2.putText(img, str_position,(0,350),font, 1, (0,250,0),4, cv2.LINE_AA)

            if ids[i]==72:
                corner4 = corners[i]
                ctr_point4 = get_center(corner4)

                # -- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
                rvec4, tvec4, markerPoints3 = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix,
                                                                           camera_distortion)
                # draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, camera_matrix, camera_distortion, rvec4, tvec4, 10)

                # print the tag position in camera frame
                str_position = "Marker position x=%4.0f y= %4.0f z=%4.0f"%(tvec4[0][0][0],tvec4[0][0][1],tvec4[0][0][2])
                cv2.putText(img, str_position,(0,400),font, 1, (0,250,0),4, cv2.LINE_AA)

        if ctr_point1 is not None and ctr_point2 is not None and ctr_point3 is not None and ctr_point4 is not None:
            # draw 4 cạnh của map
            cv2.line(img,ctr_point1,ctr_point2,(0,0,255), 5)
            cv2.line(img,ctr_point2,ctr_point3,(0,0,255), 5)
            cv2.line(img,ctr_point3,ctr_point4,(0,0,255), 5)
            cv2.line(img,ctr_point4,ctr_point1,(0,0,255), 5)
            # tính chiều dài và rộng của sa bàn
            distance = distance_between_tow_points(tvec1[0][0],tvec2[0][0])
            print("distance must be around 3: ", str(distance))
            print("datatype of ctr_point ", type(ctr_point1))
            ctr1 = np.asarray(ctr_point1, dtype=np.int32)
            ctr2 = np.asarray(ctr_point2, dtype=np.int32)
            ctr3 = np.asarray(ctr_point3, dtype=np.int32)
            ctr4 = np.asarray(ctr_point4, dtype=np.int32)

            ctr1 = np.expand_dims(ctr1,axis=0)
            ctr2 = np.expand_dims(ctr2,axis=0)
            ctr3 = np.expand_dims(ctr3,axis=0)
            ctr4 = np.expand_dims(ctr4,axis=0)

            # save tọa độ pixel của 4 coner
            position_pixel = np.concatenate((ctr1,ctr2,ctr3,ctr4),axis=0)
            np.savetxt("position_pixel_map.txt",position_pixel,delimiter=',')
            #
            distance = np.array([[round(distance,4)]])
            np.savetxt("distance_map.txt",distance,delimiter=',')

            # save tọa độ trong k gian của 3 coner
            # tvec1 = tvec1.reshape((3,0))
            position_xyz = np.concatenate((tvec1[0], tvec2[0],tvec3[0]))
            print("hello ", position_xyz)
            np.savetxt("position_xyz_map.txt",position_xyz, delimiter=",")
            print("......the process finissed, check position_map.txt, distance_map.txt, press q")


    # display the frame
    cv2.line(img, (320, 240), (390, 240), (0, 0, 255), 5)
    cv2.line(img, (320, 240), (320, 310), (0, 255, 0), 5)

    cv2.imshow('frame', img)

    # use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
