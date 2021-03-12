from TagsInfo.Global_variable import *
from math import *
import numpy as np
import cv2
from TagsInfo.tools import *
import time
from ourtag2 import *


class Tag:
    def __init__(self, points, disToSurface, rotation):
        '''
        :param points:此tag四个角点的三维坐标，二维4*3列表
        :param translation: 此tag平移矩阵
        :param rotation: 此tag旋转矩阵
        '''
        self.points = points
        self.disToSurface = disToSurface
        self.rotation = rotation

    def __str__(self):
        return "points:\n{0}\ndisToSurface:\n{1}\nrotation:\n{2}\n".format(self.points, self.disToSurface,
                                                                           self.rotation)


def getROI(img):
    '''img.shape:(2048*2592),设定2048为行数'''
    resize_num = 2
    imgresize = cv2.resize(img, ((int)(img.shape[1] / resize_num), (int)(img.shape[0] / resize_num)))
    image_size = imgresize.shape
    # decoder1 = decoder(imgresize, image_size, EnableHomography=True)
    # decoder1.detect()
    # print(decoder1.corners)
    # if len(detectroi)!=0:
    #     print("detectroi:",detectroi[0].corners)
    list = []
    maxpixhang = 0
    maxpixlie = 0
    minpixhang = 2047
    minpixlie = 2591
    # for i in range(len(decoder1.datas)):
    #     for j in range(5):
    #         maxpixhang = max(maxpixhang, decoder1.corners[i][j][1])
    #         maxpixlie = max(maxpixlie, decoder1.corners[i][j][0])
    #         minpixhang = min(minpixhang, decoder1.corners[i][j][1])
    #         minpixlie = min(minpixlie, decoder1.corners[i][j][0])
    if minpixlie != 2591:
        list = [[max((int)(minpixhang * 2 - 5), 0), max((int)(minpixlie * 2 - 5), 0)],
                [min((int)(maxpixhang * 2 + 5), 2047), min((int)(maxpixlie * 2 + 5), 2591)]]
    else:
        list = [[maxpixhang, maxpixlie], [minpixhang, minpixlie]]
    return list


def get3d(t1, t2, intr, B, tagIndex):
    '''
    :param t1: 图一某tag四个角点的亚像素坐标
    :param t2: 图二某tag四个角点的亚像素坐标
    :param intr: 相机内参（此处取左相机内参）
    :param B: baseline
    :return: points3d为四个角点三维坐标
    https://blog.csdn.net/u010368556/article/details/59647848
    双目视差法
    '''
    t1 = np.array(t1)
    t2 = np.array(t2)
    points3d = []
    tempZ0 = intr[0][0] * B / (t1 - t2)[0][0]
    global ave, avepoints, jittycnt, count, depthlistpoints, depthlist, subPointsCntL, subPointsCntR
    if (fabs(tempZ0 - ave[tagIndex]) * 1000) > 0.08:
        # print(ave[tagIndex])
        jittycnt[tagIndex] += 1
        if jittycnt[tagIndex] >= 2:
            jittycnt[tagIndex] = 0
            print("here>0.1")
            subPointsListL[tagIndex].clear()
            subPointsListR[tagIndex].clear()
            subPointsCntR[tagIndex] = 0
            subPointsCntL[tagIndex] = 0
            for i in range(4):
                depthlistpoints[tagIndex][i].clear()
            count[tagIndex] = 1
            ave[tagIndex] = tempZ0
            depthlist[tagIndex].clear()
            depthlist[tagIndex].append(tempZ0)
        else:
            # jittycnt[tagIndex] = 0
            if count[tagIndex] >= 30:
                del depthlist[tagIndex][0]
                depthlist[tagIndex].append(tempZ0)
                ave[tagIndex] = sum(depthlist[tagIndex]) / count[tagIndex]
            else:
                depthlist[tagIndex].append(tempZ0)
                ave[tagIndex] = sum(depthlist[tagIndex]) / (count[tagIndex] + 1)
                count[tagIndex] += 1

    for i in range(4):
        tempZ = intr[0][0] * B / (t1 - t2)[i][0]
        # print(depthlistpoints[tagIndex][i])
        if len(depthlistpoints[tagIndex][i]) >= 30:
            del depthlistpoints[tagIndex][i][0]
            depthlistpoints[tagIndex][i].append(tempZ)
            avepoints[tagIndex][i] = sum(depthlistpoints[tagIndex][i]) / (len(depthlistpoints[tagIndex][i]))
        else:
            depthlistpoints[tagIndex][i].append(tempZ)
            avepoints[tagIndex][i] = sum(depthlistpoints[tagIndex][i]) / (len(depthlistpoints[tagIndex][i]))
        Z = avepoints[tagIndex][i]
        X = (t1[i][0] - intr[0][2]) * Z / intr[0][0]
        Y = -(t1[i][1] - intr[1][2]) * Z / intr[1][1]
        points3d.append((X, Y, Z))
        # print(Z)
    return points3d


def getTagInfo(img, intr, TAG_SIZE, flag):
    '''
    :param img:图像
    :param intr:相机内参
    :param TAG_SIZE:tag尺寸
    :param flag:0代表左图，1代表右图
    :return: tags保存多个tag的四个角点的亚像素坐标，三维列表
    '''
    global subPointsListL, subPointsListR, aveSubPointsListL, aveSubPointsListR, subPointsCntL, subPointsCntR, leftCameraJittyList, rightCameraJittyList, subaveleft, subaveright, idDict
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    ROI = getROI(gray)
    tags = []
    tidlist = []
    # detection = []
    fidlist = []

    '''len(gray)=2048'''
    roi_before_equalized = gray
    image_size = roi_before_equalized.shape
    decoder1 = decoder(roi_before_equalized, image_size, EnableHomography=True)
    decoder1.detect()
    print(decoder1.id)
    for numm in range(len(decoder1.corners)):
        if decoder1.id[numm] not in idDict:
            k = len(idDict)
            idDict[decoder1.id[numm]] = k  # idDict存tag_id
        tidlist.append(decoder1.id[numm])
        # print(detectroi[numm].tag_id)
        startarr = np.array([ROI[0][1], ROI[0][0]])
        # detectroi[numm].center += startarr
        for i in range(5):
            decoder1.corners[numm][i] += startarr
        # print("detect in:",detectroi[numm].corners)
        # detection.append(decoder1[numm])
    detnum = tidlist[:]
    tidlist.sort()
    # print(detnum)
    for tid in detnum:
        m = detnum.index(tid)
        points = decoder1.corners[m].astype('int')
        subPoints = points.astype(np.float32)
        cv2.cornerSubPix(gray, subPoints, (5, 5), (-1, -1), criteria)
        subpixwinsize = 10
        subpixthresh = 0.13
        # print(type(subPoints))
        if (flag == 0):
            if (subPoints[0][0] - subaveleft[tid] > subpixthresh):
                leftCameraJittyList[tid] += 1
                if leftCameraJittyList[tid] >= 3:
                    print("l>0.1")
                    leftCameraJittyList[tid] = 0
                    subPointsCntL[tid] = 1
                    subPointsCntR[tid] = 0
                    subPointsListL[tid].clear()
                    subPointsListR[tid].clear()
                    subPointsListL[tid].append(subPoints)
                    subaveleft[tid] = subPoints[0][0]
                    if not fidlist.__contains__(decoder1.id[m]):
                        tags.append(subPointsListL[tid][-1].tolist())
                        fidlist.append(decoder1.id[m])
                else:
                    # tidlist.remove(tid)
                    # print('left tid:',tid)
                    if len(subPointsListL[tid]) != 0 and False == fidlist.__contains__(decoder1.id[m]):
                        tags.append(subPointsListL[tid][-1].tolist())
                        fidlist.append(decoder1.id[m])

            else:
                ave = []
                subPointsListL[tid].append(subPoints)
                # print(subPoints)
                leftCameraJittyList[tid] = 0
                if subPointsCntL[tid] < subpixwinsize:
                    subPointsCntL[tid] += 1
                    for i in range(len(subPoints)):
                        ave.append([])
                        for j in range(len(subPoints[i])):
                            sum = 0
                            for k in range(len(subPointsListL[tid])):
                                sum += subPointsListL[tid][k][i][j]
                            ave[i].append(sum / subPointsCntL[tid])
                else:
                    del subPointsListL[tid][0]
                    for i in range(len(subPoints)):
                        ave.append([])
                        for j in range(len(subPoints[i])):
                            sum = 0
                            for k in range(len(subPointsListL[tid])):
                                sum += subPointsListL[tid][k][i][j]
                            ave[i].append(sum / subPointsCntL[tid])

                subaveleft[tid] = ave[0][0]
                if not fidlist.__contains__(decoder1.id[m]):
                    tags.append(ave)
                    fidlist.append(decoder1.id[m])
        # -----------------------------------
        else:
            if (subPoints[0][0] - subaveright[tid] > subpixthresh):
                rightCameraJittyList[tid] += 1
                if rightCameraJittyList[tid] >= 3:
                    print("r>0.1")
                    rightCameraJittyList[tid] = 0
                    subPointsCntR[tid] = 1
                    subPointsCntL[tid] = 0
                    subPointsListR[tid].clear()
                    subPointsListL[tid].clear()
                    subPointsListR[tid].append(subPoints)
                    subaveright[tid] = subPoints[0][0]

                    if not fidlist.__contains__(decoder1.id[m]):
                        tags.append(subPointsListR[tid][-1].tolist())
                        fidlist.append(decoder1.id[m])
                else:
                    if len(subPointsListR[tid]) != 0 and False == fidlist.__contains__(decoder1.id[m]):
                        tags.append(subPointsListR[tid][-1].tolist())
                        fidlist.append(decoder1.id[m])
                    # print("tidlist:",tidlist)
                    # tidlist.remove(tid)
                    # print('right tid:',tid)
                    # print("tidlist:",tidlist)

            else:
                ave = []
                rightCameraJittyList[tid] = 0
                subPointsListR[tid].append(subPoints)
                if subPointsCntR[tid] < subpixwinsize:
                    subPointsCntR[tid] += 1
                    for i in range(len(subPoints)):
                        ave.append([])
                        for j in range(len(subPoints[i])):
                            sum = 0
                            for k in range(len(subPointsListR[tid])):
                                sum += subPointsListR[tid][k][i][j]
                            ave[i].append(sum / subPointsCntR[tid])
                else:
                    del subPointsListR[tid][0]
                    for i in range(len(subPoints)):
                        ave.append([])
                        for j in range(len(subPoints[i])):
                            sum = 0
                            for k in range(len(subPointsListR[tid])):
                                sum += subPointsListR[tid][k][i][j]
                            ave[i].append(sum / subPointsCntR[tid])
                subaveright[tid] = ave[0][0]
                if not fidlist.__contains__(decoder1.id[m]):
                    tags.append(ave)
                    fidlist.append(decoder1.id[m])
    return tags, fidlist  # ,tidlist


def getTags(img1, img2, P1, P2, map1x, map1y, map2x, map2y, B=0.4335, TAG_SIZE=0.029):
    # 0.4289
    '''
    :param pic1: 图一路径
    :param pic2: 图二路径
    :param camera_matrix1:左相机内参，3*3矩阵
    :param camera_matrix2:右相机内参，3*3矩阵
    :param dist_coeffs1:左相机畸变参数，1*5矩阵
    :param dist_coeffs2:右相机畸变参数，1*5矩阵
    :param R:外参，3*3旋转矩阵
    :param T:外参，1*3平移矩阵
    :param B:baseline，左右相机光心距离
    :param TAG_SIZE:tag尺寸
    :return:一个列表，内含两个Tag类对象
    '''

    # 畸变校正和立体校正
    rectifyed_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_AREA)
    # cv2.imwrite("r1.bmp",rectifyed_img1)
    # cv2.imwrite("r2.bmp",rectifyed_img2)
    tagsL, tidL = getTagInfo(rectifyed_img1, P1, TAG_SIZE, 0)
    tagsR, tidR = getTagInfo(rectifyed_img2, P2, TAG_SIZE, 1)
    # 获取到的id和tag的数量不相同
    # print('------------------------------')
    # print("tidl:", tidL)
    # print("tidR:", tidR)
    # print("tagsL:", tagsL)
    # print("tagsR:", tagsR)
    # if len(tagsR) != 0:
    #     print("type tagsR", type(tagsR[0]))
    # print('------------------------------')

    '''tagsL返回的是四个点的坐标'''
    tags = []
    tidF=set([])
    if tidL != tidR or len(tagsL) != len(tagsR):
        tidF = set(tidL) & set(tidR)
        for num in list(set(tidL) - tidF):
            index = tidL.index(num)
            tidL.remove(tidL[index])
            if len(tagsL) > 0:
                tagsL.remove(tagsL[index])
        for num in list(set(tidR) - tidF):
            index = tidR.index(num)
            tidR.remove(tidR[index])
            if len(tagsR) > 0:
                tagsR.remove(tagsR[index])

        # return 3, "两相机拍摄的tag不同，无法配对"
    if len(tagsL) == 0:
        return 0, "没有拍摄到tag"
    else:
        # print(len(tidL), len(tidR), len(tagsL), len(tagsR))
        for i in range(len(tagsL)):
            # 通过两幅图像各自tag的像素坐标通过三角关系进行三维重建，得到角点的三维坐标，封装在points3d中
            points3d = get3d(tagsL[i], tagsR[i], P1, B, tidL[i])
            '''返回四个角点的三维坐标'''
            # print("point3d",i," :",points3d)
            tagsL[i] = np.array(tagsL[i])
            points3d = np.array(points3d)
            # 得到位姿
            xAxis=points3d[1]-points3d[0]
            yAxis=points3d[3]-points3d[0]
            zAxis=np.cross(xAxis, yAxis)
            # print(xAxis,yAxis,zAxis)
            # print(points3d)
            # success, rotation, translation,inliers = cv2.solvePnPRansac(points3d, tagsL[i], camera_matrix1, dist_coeffs1,flags=cv2.SOLVEPNP_ITERATIVE
            tags.append(Tag(points3d, getDisToSurface(points3d[0], points3d[1], points3d[2]),
                            getangle(xAxis,yAxis,zAxis)))
            if testangledist['flagangle']==True:
                print("testangle")
                testangledist['xnow'] = xAxis
                testangledist['ynow'] = yAxis
                testangledist['znow'] = zAxis
                testangledist['flagangle'] = False
                testangledist['flagangle2']=True
            if testangledist['flagangle2']==True:
                h = np.array([[getcos(testangledist['xnow'], xAxis), getcos(testangledist['xnow'], yAxis), getcos(testangledist['xnow'], zAxis)],
                              [getcos(testangledist['ynow'], xAxis), getcos(testangledist['ynow'], yAxis), getcos(testangledist['ynow'], zAxis)],
                              [getcos(testangledist['znow'], xAxis), getcos(testangledist['znow'], yAxis), getcos(testangledist['znow'], zAxis)]])
                dst,jacobian=cv2.Rodrigues(h)
                # print(dst)
                dst2=np.linalg.norm(dst)
                print(dst2/math.pi*180)
            # print("-------------------------------------------------")
            # for i in range(4):
            #     print("length:"+str(i), 100 * sqrt(
            #     (tags[0].points[i][2] - tags[0].points[(i+1)%4][2]) * (tags[0].points[i][2] - tags[0].points[(i+1)%4][2]) +
            #     (tags[0].points[i][1] - tags[0].points[(i+1)%4][1]) * (tags[0].points[i][1] - tags[0].points[(i+1)%4][1]) +
            #     (tags[0].points[i][0] - tags[0].points[(i+1)%4][0]) * (tags[0].points[i][0] - tags[0].points[(i+1)%4][0])))
        # print(tags[0].points[0][2])
        return 1, tags, tidL  # '''[0].points[0][2] * 1000'''


def getDisToSurface(p1, p2, p3):
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]
    a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d = a * x1 + b * y1 + c * z1
    dis = fabs(d / (sqrt(a * a + b * b + c * c)))
    return dis * 1000


def getVector(p1, p2):
    x1, y1, z1, x2, y2, z2 = p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]
    vector = [x2 - x1, y2 - y1, z2 - z1]
    return vector


def getRotation(p1, p2):
    x1, y1, z1, x2, y2, z2 = p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]
    a, b, c = x2 - x1, y2 - y1, z2 - z1
    # vector=[x2-x1,y2-y1,z2-z1]
    length = sqrt(a * a + b * b + c * c)
    xCos = a / length
    yCos = b / length
    zCos = c / length
    xTempAngle = np.arccos(xCos)
    yTempAngle = np.arccos(yCos)
    zTempAngle = np.arccos(zCos)
    xAngle = xTempAngle * 360 / 2 / np.pi
    yAngle = yTempAngle * 360 / 2 / np.pi
    zAngle = zTempAngle * 360 / 2 / np.pi
    return [xAngle, yAngle, zAngle]


def getAngle(vector1, vector2):
    x = np.array(vector1)
    y = np.array(vector2)
    # 两个向量
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    # 相当于勾股定理，求得斜线的长度
    cos_angle = x.dot(y) / (Lx * Ly)
    # 求得cos_sita的值再反过来计算，绝对长度
    # .乘以cos角度为矢量长度，初中知识。。
    # print(cos_angle)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2
    # 变为角度
    # print(angle2)
    # x.dot(y) =  y=∑(ai*bi)
