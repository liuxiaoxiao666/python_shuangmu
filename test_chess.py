import cv2
import numpy as np
import csv
f = open('t_ches4.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
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
map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, (width, height), cv2.CV_32FC1)
data=[]
for j in range(25):
    img = cv2.imread("L/"+str(j)+".bmp")
    rectifyed_img1 = cv2.remap(img, map1x, map1y, cv2.INTER_AREA)
    gray = cv2.cvtColor(rectifyed_img1, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None) #得到棋盘格角点坐标
    # print(corners.shape)
    img2 = cv2.imread("R/"+str(j)+".bmp")
    rectifyed_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_AREA)
    gray2 = cv2.cvtColor(rectifyed_img2, cv2.COLOR_BGR2GRAY)
    size2 = gray2.shape[::-1]
    ret2, corners2 = cv2.findChessboardCorners(gray2, (11, 8), None) #得到棋盘格角点坐标
    # print(corners2)
    t1 = np.array(corners)
    t2 = np.array(corners2)
    points3d = []
    data1=[]
    for i in range(len(t1)):
        Z = 1000*P1[0][0] * B / (t1 - t2)[i][0][0]
        X = (t1[i][0][0] - P1[0][2]) * Z / P1[0][0]
        Y = -(t1[i][0][1] - P1[1][2]) * Z / P1[1][1]
        points3d.append((X, Y, Z))
    print(points3d)
    data1.append(str(points3d[0][2]))
    # print(t1)
    for i in range(len(points3d)-1):
        d=((points3d[i+1][0]-points3d[i][0])**2+(points3d[i+1][1]-points3d[i][1])**2+(points3d[i+1][2]-points3d[i][2])**2)**0.5
        data1.append(str(d))
    data.append(data1)

for i in range(len(data[0])):
    data2 = []
    for j in range(len(data)):
        data2.append(data[j][i])

    csv_writer.writerow(data2)