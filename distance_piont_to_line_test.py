# import cv2
# import numpy as np
# import random
# import cv2.aruco as aruco
# from time import time
# from utilis import*
# from datetime import datetime
# from time import sleep
#
# # reduce the brightness
# def reduce_brightness(image,intensity_value=100):
#     intensity_value /= 100
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hsv[...,2] = hsv[...,2]*intensity_value
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return img
#
# # Checks if a matrix is a valid rotation matrix.
# def isRotationMatrix(R):
#     Rt = np.transpose(R)
#     shouldBeIdentity = np.dot(Rt, R)
#     I = np.identity(3, dtype=R.dtype)
#     n = np.linalg.norm(I - shouldBeIdentity)
#     return n < 1e-6
#
#
# # Calculates rotation matrix to euler angles
# # The result is the same as MATLAB except the order
# # of the euler angles ( x and z are swapped ).
# def rotationMatrixToEulerAngles(R):
#     assert (isRotationMatrix(R))
#
#     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#
#     singular = sy < 1e-6
#
#     if not singular:
#         x = math.atan2(R[2, 1], R[2, 2])
#         y = math.atan2(-R[2, 0], sy)
#         z = math.atan2(R[1, 0], R[0, 0])
#     else:
#         x = math.atan2(-R[1, 2], R[1, 1])
#         y = math.atan2(-R[2, 0], sy)
#         z = 0
#
#     return np.array([x, y, z])
#
# ############################################################
# # set up data train conf
# label = 75                       # cm
# file = open("train_data.txt",'a')
# def save_train_data(x=0,y=0, flag = True):
#     if x > 0 and y > 0 and flag:
#         x = str(round(x,4))
#
#
#
#         y = str(y)
#         str_data = x+ ',' + y
#         file.write(str_data +'\n')
# #############################################################
# #define tag
# marker_size = 15
#
# # get camera calibration path
# camera_matrix = np.loadtxt('cameraMatrix.txt', delimiter=',')
# camera_distortion = np.loadtxt('cameraDistortion.txt', delimiter=',')
#
# # 180 deg rotation matrix around the x axis
# R_flip = np.zeros((3,3), dtype=np.float32)
# R_flip[0,0] =  1.0
# R_flip[1,1] = -1.0
# R_flip[2,2] = -1.0
#
#
# # define aruco dictionary
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
# parameters = aruco.DetectorParameters_create()
#
# ######################### get map infor #################################
# cornerPoint1 = np.zeros((1,2))
# cornerPoint2 = np.zeros((1,2))
# cornerPoint3 = np.zeros((1,2))
# cornerPoint4 = np.zeros((1,2))
# positions_map = np.loadtxt('position_pixel_map.txt', delimiter=',')
# cornerPoint1, cornerPoint2, cornerPoint3, cornerPoint4 = positions_map.astype(int)
# points = np.loadtxt('poistion_xyz_map.txt',delimiter=',') # contain 3 point positions in real 3D
# point1_coor, point2_coor, point3_coor = points
# print("point1_coor shape : ",point1_coor.shape)
# # corner1_xyz, corner2_xyz = np.loadtxt('poistion_xyz_map.txt',delimiter=',')
# ######################## get camera parameter ###########################
# # ..... capture the video
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# flag = True
# if flag:
#     brightness, contrast, saturation, gain, exposure, white_balance, focus = np.loadtxt('camera_parameter.txt', delimiter=',').astype(int)
#     # cap.set(cv2.CAP_PROP_EXPOSURE, 15)
#     # set camera size
#
#     cap.set(10, brightness) # brightness     min: 0   , max: 255 , increment:1
#     cap.set(11, contrast) # contrast       min: 0   , max: 255 , increment:1
#     cap.set(12, saturation) # saturation     min: 0   , max: 255 , increment:1
#     # cap.set(13, 13) # hue
#     cap.set(14, gain) # gain           min: 0   , max: 127 , increment:1
#     cap.set(15, exposure) # exposure       min: -7  , max: -1  , increment:1
#     cap.set(17, white_balance) # white_balance  min: 4000, max: 7000, increment:1
#     cap.set(28, focus) # focus          min: 0   , max: 255 , increment:5
# font = cv2.FONT_HERSHEY_PLAIN
# ###################################################################################
# # modify the brightness through HSV
# def nothing():
#     pass
# cv2.namedWindow('Trackbars')
# cv2.resizeWindow('Trackbars',650,250)
# cv2.createTrackbar('Brightness','Trackbars',0,100,nothing)
# brightness_hsv = 100
# ###################################################################################
# def get_and_save_moveInfor(speed, x_coor, y_coor):
#     """
#
#     :return: corodinate of car in map
#     """
#     file = open('main_data.txt','a')
#     now = datetime.now()
#     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#     file.write(dt_string+','+ str(speed)+','+str(x_coor)+','+str(y_coor)+'\n')
#     file.close()
# ###################################################################################
# c_time = 0
# p_time = 0
#
# run = True
# temp = 0
# temp_tvec = np.array([0,0,0])
# while run:
#
#     ret, img = cap.read()
#     img = reduce_brightness(img, brightness_hsv)
#     brightness_hsv = cv2.getTrackbarPos('Brightness', 'Trackbars')
#     # convert to gray scale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # find all the aruco maker in the image
#     corners, ids, rejected = aruco.detectMarkers(image= gray, dictionary=aruco_dict, parameters = parameters,
#                                                 cameraMatrix = camera_matrix, distCoeff = camera_distortion)
#
#     # fps
#     c_time = time()
#     fps = round(1/(c_time-p_time),1)
#
#     # ##
#     tvec1 = None
#     tvec2 = None
#     tvec3 = None
#     if len(corners) != 0 :
#         for i in range(len(ids)):
#             if ids[i] == 75:
#                 #-- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
#                 rvec,tvec1, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, camera_distortion)
#
#                 # draw the detected marker and put a reference frame over it
#                 aruco.drawDetectedMarkers(img,corners)
#                 aruco.drawAxis(img, camera_matrix, camera_distortion, rvec, tvec1, 5)
#
#                 #############
#                 tvec1 = tvec1.reshape((3,))
#
#             if ids[i] == 25:
#                 #-- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
#                 rvec,tvec2, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, camera_distortion)
#
#                 # draw the detected marker and put a reference frame over it
#                 aruco.drawDetectedMarkers(img,corners)
#                 aruco.drawAxis(img, camera_matrix, camera_distortion, rvec, tvec2, 5)
#
#                 #############
#                 tvec2 = tvec2.reshape((3,))
#
#             if ids[i] == 35:
#                 #-- res = [rvec,tvec, ?]  array of rotation and position of each marker in camera drame
#                 rvec,tvec3, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, camera_distortion)
#
#                 # draw the detected marker and put a reference frame over it
#                 aruco.drawDetectedMarkers(img,corners)
#                 aruco.drawAxis(img, camera_matrix, camera_distortion, rvec, tvec3, 5)
#
#                 #############
#                 tvec3 = tvec3.reshape((3,))
#
#
#     if tvec1 is not None and tvec2 is not None and tvec3 is not None:
#
#         # project_point = tvec
#         x_coor = distance_point_to_line(tvec1, tvec2,tvec3)
#
#         print('Distance :', str(x_coor))
#         # get_and_save_moveInfor(speed,x_coor,y_coor)
#         # sleep(0.1)
#
#     p_time = c_time
#     # display the frame
#     cv2.putText(img, "fps :"+str(fps), [10,20],cv2.FONT_HERSHEY_PLAIN,2,[0,255,255],2)
#     cv2.imshow('frame', img)
#
#     # use 'q' to quit
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# file.close()
#
#
#
#

import pandas as pd
import numpy as np

data = pd.read_csv('C:\Users\Admin\Downloads\ars wind power forcast\data\sdwpf_baidukddcup2022_full.CSV')
data_np = data.values
np.save('hola.npy', data_np)

