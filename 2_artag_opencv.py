import numpy as np
import cv2
import glob


# SET THE PARAMETER
nRows = 9
nCols = 6
dementions = 24 # mm

# termination criteria
creteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, dementions, 0.001 )

# prepare object points
objp = np.zeros((nRows*nCols,3), np.float32)
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image_path = 'save_images' + "/*" + ".jpg"
images = glob.glob(image_path)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), creteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nCols,nRows), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

imgNotGood = "save_images/snapshot_1920_1080_0.jpg"

img = cv2.imread(imgNotGood)
h, w = img.shape[:2]
print("Image to undistort: ", imgNotGood)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
print("ROI: ", x, y, w, h)

# cv2.imwrite("calibresult.png", dst)
print("Calibrated picture saved as calibresult.png")
print("Calibration Matrix: ")
print(mtx)
print("Disortion: ", dist)

filename = "cameraMatrix.txt"
np.savetxt(filename, mtx, delimiter=',')
filename = "cameraDistortion.txt"
np.savetxt(filename, dist, delimiter=',')

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))

