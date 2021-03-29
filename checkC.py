import cv2
import numpy as np
camera_matrix1 = np.array([[3.54146051e+03, 0.00000000e+00, 2.02736891e+03],
 [0.00000000e+00 ,3.54195528e+03, 1.52639175e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_matrix2 = np.array([[3.54882867e+03, 0.00000000e+00 ,2.05877736e+03],
 [0.00000000e+00 ,3.54768120e+03, 1.54107730e+03],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

dist_coeffs1 = np.array(
    [-0.09569434,  0.10140725 , 0.00025152, -0.000199 ,  -0.0185507 ])
dist_coeffs2 = np.array(
    [-9.72027380e-02 , 1.07699683e-01 , 4.77391537e-04,  1.04441111e-04,
     -2.92863749e-02])
R = np.array([[ 0.98542502, -0.00680529 , 0.16997419],
 [ 0.00627234  ,0.99997359 , 0.00367226],
 [-0.16999469 ,-0.0025526 ,  0.98544167]])
T = np.array([-0.20809611, -0.00026478, 0.0190811])
width = 4096
height = 3000
B=0.20896925487446831588752564649734
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
                                                      (width, height), R, T, alpha=1)
print(P1,P2)
map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, (width, height), cv2.CV_32FC1)
# 畸变校正和立体校正
pic1 = cv2.imread('17L.bmp')
pic2 = cv2.imread('17R.bmp')
rectifyed_img1 = cv2.remap(pic1, map1x, map1y, cv2.INTER_AREA)
rectifyed_img2 = cv2.remap(pic2, map2x, map2y, cv2.INTER_AREA)
cv2.imwrite("img19.bmp", rectifyed_img1)
cv2.imwrite("img20.bmp", rectifyed_img2)