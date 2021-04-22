from TagsInfo.Global_variable import *
from math import *
import numpy as np
import cv2
from TagsInfo.tools import *
import time
import pandas as pd
import os
class Tag:
    def __init__(self, points, disToSurface, rotation,R):
        '''
        :param points:此tag四个角点的三维坐标，二维4*3列表
        :param disToSurface:左相机到标签平面的距离
        :param translation: 此tag平移矩阵,定义某一个点的位置
        :param rotation: 此tag绕三个轴的旋转度
        :param R:此tag旋转矩阵
        '''
        self.points = points
        self.disToSurface = disToSurface  # 未启用
        self.rotation = rotation
        self.translation=points[0]
        self.R=R

    def __str__(self):
        return "points:\n{0}\ndisToSurface:\n{1}\nrotation:\n{2}\nrotation:\n{3}\nrotation:\n{4}\n".format(self.points, self.disToSurface,
                                                                           self.rotation,self.translation,self.R)


def getROI(img): #返回全图
    resize_num = 2
    imgresize = cv2.resize(img, ((int)(img.shape[1] / resize_num), (int)(img.shape[0] / resize_num)))
    detectroi = ap_detector.detect(imgresize)
    (hang,lie)=img.shape
    hang-=1
    lie-=1
    maxpixhang = 0
    maxpixlie = 0
    minpixhang = hang
    minpixlie = lie
    for i in range(len(detectroi)):
        for j in range(4):
            maxpixhang = max(maxpixhang, detectroi[i].corners[j][1])
            maxpixlie = max(maxpixlie, detectroi[i].corners[j][0])
            minpixhang = min(minpixhang, detectroi[i].corners[j][1])
            minpixlie = min(minpixlie, detectroi[i].corners[j][0])
    rangh=(maxpixhang-minpixhang)//2
    rangl=(maxpixlie-minpixlie)//2
    if minpixlie != lie:
        list = [[max((int)(minpixhang- rangh) * resize_num , 0), max((int)(minpixlie - rangh)* resize_num , 0)],
                [min((int)(maxpixhang+ rangl) * resize_num, hang), min((int)(maxpixlie+ rangl) * resize_num , lie)]]
    else:
        list = [[maxpixhang, maxpixlie], [minpixhang, minpixlie]]
    # print(list)
    return list


def get3d(t1, t2, intr, B, tagIndex): #计算四个角点三维坐标，深度滤波
    '''
    :param t1: 图一某tag四个角点的亚像素坐标
    :param t2: 图二某tag四个角点的亚像素坐标
    :param intr: 相机内参（此处取左相机内参）
    :param B: baseline
    :return: points3d为四个角点三维坐标
    '''
    t1 = np.array(t1)
    t2 = np.array(t2)
    points3d = []
    tempZ0 = intr[0][0] * B / (t1 - t2)[0][0]
    avewinsize=60
    global ave, avepoints, jittycnt, count, depthlistpoints, depthlist, subPointsCntL, subPointsCntR
    if (fabs(tempZ0 - ave[tagIndex]) * 1000) > 0.14:
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
            if count[tagIndex] >= avewinsize:
                del depthlist[tagIndex][0]
                depthlist[tagIndex].append(tempZ0)
                ave[tagIndex] = sum(depthlist[tagIndex]) / count[tagIndex]
            else:
                depthlist[tagIndex].append(tempZ0)
                ave[tagIndex] = sum(depthlist[tagIndex]) / (count[tagIndex] + 1)
                count[tagIndex] += 1

    for i in range(4):
        tempZ = intr[0][0] * B / (t1 - t2)[i][0]
        if len(depthlistpoints[tagIndex][i]) >= avewinsize:
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
    return points3d


def getTagInfo(img, flag,R_now,P_now):
    '''
    :param img:图像
    :param intr:相机内参
    :param TAG_SIZE:tag尺寸
    :param flag:0代表左图，1代表右图
    :return: tags保存多个tag的四个角点的亚像素坐标，三维列表
    '''
    global subPointsListL, subPointsListR, aveSubPointsListL, aveSubPointsListR, subPointsCntL,\
        subPointsCntR, leftCameraJittyList, rightCameraJittyList, subaveleft, subaveright, idDict,roi_p

    if len(img.shape)==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray=img
    # print(gray)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    detection = []
    startpix = [0, 0]
    subareas = []
    if roi_p["all_view"] == False:
        # print("jubu")
        if flag == 0:
            subareas = roi_p["tagsinfol"]
        else:
            subareas = roi_p["tagsinfor"]
    else:  # 返回全图
        # print("quanju")
        subareas = [getROI(gray)]
    tags = []
    tags_raw=[]
    tidlist = []
    fidlist = []
    for i in range(len(subareas)):
        '''len(gray)=2048'''
        ROI = subareas[i]

        roi_before_equalized = gray[int(ROI[0][0]):int(ROI[1][0] + 1), int(ROI[0][1]):int(ROI[1][1] + 1)]
        # print(ROI)
        # if flag==0:
        #
        #     cv2.imwcv2.imwrite("img7.bmp", roi_before_equalized)
        # else:
        #     cv2.imwrite("img8.bmp", roi_before_equalized)
        try:
            # nt['nt'] = time.time()
            detectroi = at_detector.detect(roi_before_equalized)
            # print(time.time() - nt['nt'])
            if flag==0:
                img_list['roi1']=roi_before_equalized
            else:
                img_list['roi2'] = roi_before_equalized
            # print("roi1 shape:", img_list['roi1'].shape)
            # print("roi2 shape:",img_list['roi2'].shape)
            for numm in range(len(detectroi)):
                if detectroi[numm].tag_id not in idDict:
                    k = len(idDict)
                    idDict[detectroi[numm].tag_id] = k  # idDict存tag_id
                tidlist.append(detectroi[numm].tag_id)
                # print(detectroi[numm].tag_id)
                startarr = np.array([ROI[0][1], ROI[0][0]])
                detectroi[numm].center += startarr
                for i in range(4):
                    detectroi[numm].corners[i] += startarr
                # print("detect in:",detectroi[numm].corners)
                detection.append(detectroi[numm])
            detnum = tidlist[:]
            tidlist.sort()
            # print(detnum)
            for tid in detnum:            #滤波
                m = detnum.index(tid)
                points = detection[m].corners
                subPoints_raw = points
                # cv2.cornerSubPix(gray, subPoints_raw, (5, 5), (-1, -1), criteria)
                subPoints_raw.shape=(4,1,2)
                if flag==0:
                    subPoints=cv2.undistortPoints(subPoints_raw,camera_matrix1,dist_coeffs1,R=R_now,P=P_now)
                else:
                    subPoints=cv2.undistortPoints(subPoints_raw, camera_matrix2, dist_coeffs2,R=R_now,P=P_now)
                # print("point:", subPoints[0])
                subPoints.shape=(-1,2)
                subPoints_raw.shape = (-1, 2)
                # print(subPoints)
                subpixwinsize = 10
                subpixthresh = 0.5
                if flag == 0:
                    if tid >= len(subaveleft):
                        continue
                    if subPoints[0][0] - subaveleft[tid] > subpixthresh:
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
                            if not fidlist.__contains__(detection[m].tag_id):
                                tags.append(subPointsListL[tid][-1].tolist())
                                fidlist.append(detection[m].tag_id)
                                tags_raw.append(subPoints_raw)
                        else:
                            # tidlist.remove(tid)
                            # print('left tid:',tid)
                            if len(subPointsListL[tid]) != 0 and False == fidlist.__contains__(detection[m].tag_id):
                                tags.append(subPointsListL[tid][-1].tolist())
                                fidlist.append(detection[m].tag_id)
                                tags_raw.append(subPoints_raw)
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
                        if not fidlist.__contains__(detection[m].tag_id):
                            tags.append(ave)
                            fidlist.append(detection[m].tag_id)
                            tags_raw.append(subPoints_raw)
                # -----------------------------------
                else:
                    if tid >= len(subaveright):
                        continue
                    if subPoints[0][0] - subaveright[tid] > subpixthresh:
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

                            if not fidlist.__contains__(detection[m].tag_id):
                                tags.append(subPointsListR[tid][-1].tolist())
                                fidlist.append(detection[m].tag_id)
                                tags_raw.append(subPoints_raw)
                        else:
                            if len(subPointsListR[tid]) != 0 and False == fidlist.__contains__(detection[m].tag_id):
                                tags.append(subPointsListR[tid][-1].tolist())
                                fidlist.append(detection[m].tag_id)
                                tags_raw.append(subPoints_raw)
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
                        if not fidlist.__contains__(detection[m].tag_id):
                            tags.append(ave)
                            fidlist.append(detection[m].tag_id)
                            tags_raw.append(subPoints_raw)
        # print("tags:",tags)
        # print("tags_raw:",tags_raw)
        except:
            pass
    return tags, fidlist, tags_raw # ,tidlist


def getTags(img1, img2,P1, P2,R1,R2):
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

    tagsL, tidL,tagsL_raw = getTagInfo(img1,  0,R1,P1)
    tagsR, tidR,tagsR_raw = getTagInfo(img2, 1,R2,P2)


    # getChessboard(img1,  0,R1,P1)
    # getChessboard(img2, 1, R2, P2)

    if ids['tl'] != []:
        if ids['tl'] != tidL or ids['tr'] != tidR:
            roi_p["all_view"] = True
        else:
            roi_p["all_view"] = False

    # print(ids['tl'],tidL)
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
            # print("point3d",tidL[i]," :",points3d)

            tagsL[i] = np.array(tagsL[i])
            points3d = np.array(points3d)
            # 得到位姿
            # print(points3d)
            pnum=0
            if testmovdist['flagmov']==True:
                print("testmove")
                if testmovdist['init_num']<100:
                    testmovdist['xsum']+=points3d[pnum][0]
                    testmovdist['ysum'] += points3d[pnum][1]
                    testmovdist['zsum'] += points3d[pnum][2]
                    testmovdist['init_num']+=1
                else:
                    testmovdist['xnow']=testmovdist['xsum']/100
                    testmovdist['ynow'] = testmovdist['ysum']/100
                    testmovdist['znow'] = testmovdist['zsum']/100
                    testmovdist['flagmov'] =False
                    testmovdist['flagmov2'] =True
            if testmovdist['flagmov2'] ==True:
                mov=1000*((points3d[pnum][0]-testmovdist['xnow'])**2+(points3d[pnum][1]-testmovdist['ynow'])**2
                          +(points3d[pnum][2]-testmovdist['znow'])**2)**0.5
                print(mov)
                print("round:",testmovdist['wd_cnt'])
                if testmovdist['save']==True:
                    testmovdist['wd_v'].append(mov)
                    if len(testmovdist['wd_v'])==testmovdist['wd_num']:
                        if os.path.exists(testmovdist['wd_fname']):
                            data = pd.read_csv(testmovdist['wd_fname'])
                            data["round:"+str(testmovdist['wd_cnt'])]=testmovdist['wd_v']

                            data.to_csv(testmovdist['wd_fname'], mode='w', index=False)
                            testmovdist['wd_cnt'] += 1
                        else:
                            with open(testmovdist['wd_fname'], 'w', newline='') as f:
                                mywrite = csv.writer(f)
                                mywrite.writerow(["round:0"])
                                for twd2 in range(testmovdist['wd_num']):
                                    mywrite.writerow([str(testmovdist['wd_v'][twd2])])
                                testmovdist['wd_cnt'] += 1
                        print("output")
                        testmovdist['wd_v'].clear()
                        # b.config(state=tk.ACTIVE)
                        testmovdist['save']= False

            xAxis=100*(points3d[1]-points3d[0])
            yAxis=100*(points3d[3]-points3d[0])
            zAxis=np.cross(xAxis, yAxis)
            if testangledist['flagangle']==True:
                print("testangle")
                if testangledist['init_num'] < 100:
                    testangledist['xsum'] += xAxis
                    testangledist['ysum'] += yAxis
                    testangledist['init_num'] += 1
                else:
                    testangledist['xnow']=testangledist['xsum']/100
                    testangledist['ynow'] = testangledist['ysum']/100
                    zAxis = np.cross(testangledist['xnow'], testangledist['ynow'])
                    testangledist['znow'] = zAxis
                    testangledist['flagangle'] = False
                    testangledist['flagangle2'] = True

            if testangledist['flagangle2']==True:
                h = np.array([[getcos(testangledist['xnow'], xAxis), getcos(testangledist['xnow'], yAxis), getcos(testangledist['xnow'], zAxis)],
                              [getcos(testangledist['ynow'], xAxis), getcos(testangledist['ynow'], yAxis), getcos(testangledist['ynow'], zAxis)],
                              [getcos(testangledist['znow'], xAxis), getcos(testangledist['znow'], yAxis), getcos(testangledist['znow'], zAxis)]])
                dst,jacobian=cv2.Rodrigues(h)
                # print(dst)
                dst2=np.linalg.norm(dst)
                ag=dst2/math.pi*180
                testangledist['agv'].append(ag)
                if len(testangledist['agv'])==1:
                    print(mean(testangledist['agv']))
                    testangledist['agv'].clear()
            angle_result=getangle(xAxis,yAxis,zAxis)
            tags.append(Tag(points3d, getDisToSurface(points3d[0], points3d[1], points3d[2]),
                            angle_result[0],angle_result[1]))
            # print("length:",100*sqrt((tags[0].points[0][2] -tags[0].points[1][2])*(tags[0].points[0][2] -tags[0].points[1][2])+
            #                      (tags[0].points[0][1] -tags[0].points[1][1])*(tags[0].points[0][1] -tags[0].points[1][1])+
            #                      (tags[0].points[0][0] -tags[0].points[1][0])*(tags[0].points[0][0] -tags[0].points[1][0])))
            # tag['tags'][1][0].points[0][1] * 1000
            # tag['tags'][1][0].points[0][0] * 1000

            # success, rotation, translation,inliers = cv2.solvePnPRansac(points3d, tagsL[i], camera_matrix1, dist_coeffs1,flags=cv2.SOLVEPNP_ITERATIVE
            # tags.append(Tag(points3d, getDisToSurface(points3d[0], points3d[1], points3d[2]),
            #                 getangle(xAxis,yAxis,zAxis)))
            # print("-------------------------------------------------")
            # for i in range(4):
            #     if i==1:
            #         print("length:"+str(i), 1000 * sqrt(
            #         (tags[0].points[i][2] - tags[0].points[(i+1)%4][2]) * (tags[0].points[i][2] - tags[0].points[(i+1)%4][2]) +
            #         (tags[0].points[i][1] - tags[0].points[(i+1)%4][1]) * (tags[0].points[i][1] - tags[0].points[(i+1)%4][1]) +
            #         (tags[0].points[i][0] - tags[0].points[(i+1)%4][0]) * (tags[0].points[i][0] - tags[0].points[(i+1)%4][0])))



            if test_area['ta_flag'] and len(test_area['ta_v'])/4 < test_area['ta_num']:
                print(len(test_area['ta_v'])/4)
                for i in range(4):
                    s_len=1000 * sqrt(
                (tags[0].points[i][2] - tags[0].points[(i+1)%4][2]) * (tags[0].points[i][2] - tags[0].points[(i+1)%4][2]) +
                (tags[0].points[i][1] - tags[0].points[(i+1)%4][1]) * (tags[0].points[i][1] - tags[0].points[(i+1)%4][1]) +
                (tags[0].points[i][0] - tags[0].points[(i+1)%4][0]) * (tags[0].points[i][0] - tags[0].points[(i+1)%4][0]))
                    test_area['ta_v'].append(s_len)
            elif test_area['ta_flag'] and len(test_area['ta_v'])/4 == test_area['ta_num']:
                if os.path.exists(test_area['ta_fname']):
                    data = pd.read_csv(test_area['ta_fname'])
                    data1=[]
                    data1.append(" y:" + str(tagpos['tags'][1][0].points[0][1] * 1000))
                    data1.append(" x:" + str(tagpos['tags'][1][0].points[0][0] * 1000))
                    data1+=test_area['ta_v']
                    data1.append(max(test_area['ta_v'])-min(test_area['ta_v']))
                    data["z:"+str(tagpos['tags'][1][0].points[0][2] * 1000)]=data1
                    data.to_csv(test_area['ta_fname'], mode='w', index=False)
                else:
                    with open(test_area['ta_fname'], 'w', newline='') as f:
                        mywrite = csv.writer(f)
                        da = []
                        da.append("z:"+str(tagpos['tags'][1][0].points[0][2] * 1000))
                        da.append(" y:" + str(tagpos['tags'][1][0].points[0][1] * 1000))
                        da.append(" x:" + str(tagpos['tags'][1][0].points[0][0] * 1000))
                        for lda in range(3):
                            mywrite.writerow([da[lda]])
                        for taa in test_area['ta_v']:
                            mywrite.writerow([str(taa)])
                        mywrite.writerow([max(test_area['ta_v'])-min(test_area['ta_v'])])
                print("output")
                test_area['ta_v'].clear()
                # b.config(state=tk.ACTIVE)
                test_area['ta_flag'] = False
                test_area['finish'] = True

        if roi_p["initnum"] < 2 and roi_p["all_view"]==True:
            roi_p["initnum"] += 1
            if len(tags) >= roi_p["tagsnum"]:
                ids['tl'] = tidL
                ids['tr'] = tidR
                roi_p["tagsnum"] = len(tags)
                roi_p["tagsinfol"].clear()
                roi_p["tagsinfor"].clear()
                for t in range(len(tags)):
                    roi_p["tagsinfol"].append(tags_2points(tagsL_raw[t]))
                    roi_p["tagsinfor"].append(tags_2points(tagsR_raw[t]))
        else:
            roi_p["all_view"] = False  #
            roi_p["initnum"] = 0
            roi_p["tagsinfol"].clear()
            roi_p["tagsinfor"].clear()
            for t in range(len(tags)):
                roi_p["tagsinfol"].append(tags_2points(tagsL_raw[t]))
                roi_p["tagsinfor"].append(tags_2points(tagsR_raw[t]))

        return 1, tags, tidL  # '''[0].points[0][2] * 1000'''

def tags_2points(tagps):
    '''
    :param tagps:图像中标签的五个角点的坐标信息
    :return: 返回能够包含各个标签的最小的正方形的边界信息
    '''
    maxpixhang = 0
    maxpixlie = 0
    minpixhang = 2999
    minpixlie = 4095
    for points in tagps:
        maxpixhang = max(maxpixhang, points[1])
        maxpixlie = max(maxpixlie,points[0])
        minpixhang = min(minpixhang, points[1])
        minpixlie = min(minpixlie, points[0])
    if minpixlie != 4095:
        addhang=(maxpixhang-minpixhang)*0.25
        addlie = (maxpixlie - minpixlie) * 0.25
        list = [[max((int)(minpixhang - addhang), 0), max((int)(minpixlie - addlie), 0)],
                [min((int)(maxpixhang + addhang), 2999), min((int)(maxpixlie + addlie), 4095)]]
    else:
        list = [[maxpixhang, maxpixlie], [minpixhang, minpixlie]]
    return list

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
