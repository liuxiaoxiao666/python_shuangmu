import cv2
import numpy as np
objp = np.zeros((11*8,3), np.float32)
objp[:,:2] = np.mgrid[0:0.1501:0.015,0:0.1051:0.015].T.reshape(-1,2)
# print(objp)
obj_points = []  # 存储3D点
img_pointsL = []  # 存储左图2D点
img_pointsR = []  # 存储右图2D点
size=(4096,3000)
for j in range(30):
    img=cv2.imread("L4/"+str(j)+".bmp")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(size)
    print(j)
    ret,corner=cv2.findChessboardCornersSB(img,(11,8),cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    obj_points.append(objp)
    img_pointsL.append(corner)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_points, img_pointsL, size, None, None)
for j in range(30):
    img=cv2.imread("R4/"+str(j)+".bmp")
    print(j)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,corner2=cv2.findChessboardCornersSB(img,(11,8),cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    # obj_points.append(objp)
    img_pointsR.append(corner2)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_points, img_pointsR, size, None, None)
stereocalibration_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(
            obj_points,img_pointsL,img_pointsR,mtxL,distL,mtxR,distR,
            size)
print(cameraMatrix1)
print(distCoeffs1)
print(cameraMatrix2)
print(distCoeffs2)
print(R)
print(T)