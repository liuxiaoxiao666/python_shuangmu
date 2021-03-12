import cv2
import numpy as np
camera_matrix1 = np.array([[3.574039860680742e+03, 1.098719182713949, 2.034292584522333e+03],
                               [0.00000000e+00, 3.572516961910460e+03, 1.506600908172405e+03],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_matrix2 = np.array([[3.577313313259017e+03, -1.129222862508354, 2.055469746090763e+03],
                           [0, 3.574454683953720e+03, 1.525514668948427e+03],
                           [0.00000000e+00, 0.00000000e+00, 1]])

dist_coeffs1 = np.array(
    [-0.092992078636370, 0.096222535649721, 4.719115037495438e-05, -2.361695408341265e-05, -0.008870832773170])
dist_coeffs2 = np.array(
    [-0.094914780676932, 0.096928513695695, 3.747002421475770e-04, 1.843273274888114e-05,-0.008310968949008])
R = np.array([[0.985055260835653,-0.005851365777969, 0.172139172237281],
              [0.005512578564103,0.999981813938738, 0.002446072232485],
              [-0.172150354567101,-0.001460585610075, 0.985069602673891]])
T = np.array([-0.2087596722021999, -0.430732180863642e-03, 0.018631547757230287])
width = 4096
height = 3000
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
                                                      (width, height), R, T, alpha=1)
map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, (width, height), cv2.CV_32FC1)
# 畸变校正和立体校正
pic1 = cv2.imread('00.bmp')
pic2 = cv2.imread('01.bmp')
rectifyed_img1 = cv2.remap(pic1, map1x, map1y, cv2.INTER_AREA)
rectifyed_img2 = cv2.remap(pic2, map2x, map2y, cv2.INTER_AREA)
cv2.imwrite("img5.bmp", rectifyed_img1)
cv2.imwrite("img6.bmp", rectifyed_img2)