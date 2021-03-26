import cv2
import numpy as np
camera_matrix1 = np.array([[3.564100914701934e+03, -5.354055265107811, 2.018475682613483e+03],
                           [0.00000000e+00, 3.562907129221029e+03, 1.517493515200955e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_matrix2 = np.array([[3.535439675347217e+03, -3.452783153381272, 2.051321308630927e+03],
                           [0, 3.532061057382973e+03, 1.539273250373704e+03],
                           [0.00000000e+00, 0.00000000e+00, 1]])

dist_coeffs1 = np.array(
    [-0.098549396542964, 0.119462724440395, 2.608416883084959e-04, 1.265176442259160e-04, -0.053361998496539])
dist_coeffs2 = np.array(
    [-0.088506610293866,0.044832527198377, -4.573903914640852e-05, -3.508585444409916e-05,0.080072156254630])
R = np.array([[0.985321244897772,-0.006303705895017, 0.170593984785791],
              [0.006274658954030,0.999980062479331, 7.094355818489540e-04],
              [-0.170595055637953,3.713971234576565e-04, 0.985341153639723]])
T = np.array([-0.2097780450660247, 0.347813027543609e-03, 0.013141942047608467])
width = 4096
height = 3000

B=0.21036229127314401357310955673791
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
cv2.imwrite("img17.bmp", rectifyed_img1)
cv2.imwrite("img18.bmp", rectifyed_img2)