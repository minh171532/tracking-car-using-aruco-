import cv2
from time import time
import numpy as np

brightness = 0
contrast = 0
saturation = 0
gain = 0
exposure = 0
white_balance = 0
focus = 0

def nothing(temp):
    print(temp)

def set_value(start_val, end_val, convert_val):
    """
    convert value from 0 - 100 to value from start_value to end_value
    :param start_val:
    :param end_val:
    :param convert_val:
    :return:
    """
    length = end_val - start_val
    return int(start_val + convert_val*length/100)

cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars',800,400)
cv2.createTrackbar('Brightness','Trackbars',0,100,nothing)
cv2.createTrackbar('contrast','Trackbars',0,100,nothing)
cv2.createTrackbar('saturation','Trackbars',0,100,nothing)
# cv2.createTrackbar('hue','Trackbars',0,255,nothing)
cv2.createTrackbar('gain','Trackbars',0,100,nothing)
cv2.createTrackbar('exposure','Trackbars',0,100,nothing)
cv2.createTrackbar('white_balance','Trackbars',0,100,nothing)
cv2.createTrackbar('focus','Trackbars',0,100,nothing)

# cv2.createTrackbar('Brightness','Trackbars',0,255,nothing)
# cv2.createTrackbar('contrast','Trackbars',0,255,nothing)
# cv2.createTrackbar('saturation','Trackbars',0,255,nothing)
# # cv2.createTrackbar('hue','Trackbars',0,255,nothing)
# cv2.createTrackbar('gain','Trackbars',0,127,nothing)
# # cv2.createTrackbar('exposure','Trackbars',-7,1,nothing)
# cv2.createTrackbar('white_balance','Trackbars',4000,7000,nothing)
# cv2.createTrackbar('focus','Trackbars',0,255,nothing)
cap = cv2.VideoCapture(0)


c_time = 0
p_time = 0

while True:
    _, img = cap.read()
    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars')
    contrast = cv2.getTrackbarPos('contrast', 'Trackbars')
    saturation = cv2.getTrackbarPos('saturation', 'Trackbars')
    gain = cv2.getTrackbarPos('gain', 'Trackbars')
    exposure = cv2.getTrackbarPos('exposure', 'Trackbars')
    white_balance= cv2.getTrackbarPos('white_balance', 'Trackbars')
    focus = cv2.getTrackbarPos('focus', 'Trackbars')
    ##########################
    brightness = set_value(0,255, brightness)
    contrast = set_value(0,255, contrast)
    saturation = set_value(0,255, saturation)
    gain = set_value(0,127,gain)
    exposure = set_value(-7,1, exposure)
    white_balance = set_value(4000,7000, white_balance)
    focus = set_value(0,255,focus)

    # fps
    c_time = time()
    fps = round(1/(c_time-p_time),1)
    p_time = c_time

    cv2.putText(img, "fps :"+str(fps), [10,20],cv2.FONT_HERSHEY_PLAIN,2,[0,255,255],2)

    cv2.imshow("asfas",img)
    cv2.waitKey(1)
    cap.set(10, brightness)  # brightness
    cap.set(11, contrast) # contrast       min: 0   , max: 255 , increment:1
    cap.set(12, saturation) # saturation     min: 0   , max: 255 , increment:1
    cap.set(14, gain) # gain           min: 0   , max: 127 , increment:1
    cap.set(15, exposure) # exposure       min: -7  , max: -1  , increment:1
    cap.set(17, white_balance) # white_balance  min: 4000, max: 7000, increment:1
    cap.set(28, focus) # focus          min: 0   , max: 255 , increment:5

    camera_parameter = np.array([[brightness, contrast, saturation, gain, exposure, white_balance, focus]])
    np.savetxt('camera_parameter.txt', camera_parameter, delimiter=',')

