import cv2
import time
import numpy as np
import pupil_apriltags as apriltag
import math

"""A detector for new tag.This program creates two classes that 
are used to detect new tags and extract information from them.
The new tag is designed based on apriltag.There are four circle
around the apriltag, circles are used as the feature points of the tags.
Using this module, you can identify all tags visiable in an image, and
get information about the location of the tags. 

Author: Dong Qifeng
Updates: Dong Qifeng,Spring 2021
Apriltags 3 version: Aleksandar Petrov, Spring 2019
"""

class Detection:
    """标签信息保存"""
    def __init__(self):
        self.tag_family = None
        self.tag_id = None
        self.center = None
        self.apcorners = None
        self.corners=None
        self.keypoints=None

    def __str__(self):
        return (
            "Detection object:"
            + "\ntag_family = "
            + str(self.tag_family)
            + "\ntag_id = "
            + str(self.tag_id)
            + "\ncenter = "
            + str(self.center)
            + "\napriltag corners = "
            + str(self.apcorners)
            + "\ncircles  = "
            +str(self.corners)
        )

    def __repr__(self):
        return self.__str__()


class Detector:
    """封装检测类"""
    def __init__(
            self,
            families="tag36h11",
            debug=False
    ):
        self._detector=apriltag.Detector(families=families)
        self._debug=debug

    def _getlength(self,a, b):
        """两个顶点a,b,返回值为线段ab的长度"""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _triangleArea(self,p1, p2, p3):
        '''计算由p1,p2,p3三个点组成的三角形的面积'''
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)

    def _isinQurd(self,a, b, c, d, e):
        '''
        :param a:四边形顶点
        :param b:四边形顶点
        :param c:四边形顶点
        :param d:四边形顶点
        :param e:待解决点
        :return: boolean值说明是否在四边形内部
        '''
        triarea = self._triangleArea(a, b, e) + self._triangleArea(b, c, e) + self._triangleArea(c, d, e) + self._triangleArea(a, d, e)
        qarea = self._triangleArea(a, b, c) + self._triangleArea(a, d, c)
        return math.fabs(triarea-qarea)<0.000001

    def _ridiusillegal(self,r, l):
        """
        :param
        :r :候选圆的直径
        :l :正方形的边长
        :return
        True: 这个圆的半径不满足要求
        """
        r = r / 2
        if r > l / 5 or r < l / 10:
            return True
        return False

    def _regeonillegal(self,a, b, c, d, edgel, keypoint):
        """
        :param a,b,c,d:四个角点
        :param edgel: 边长的均值
        :param keypoint: 候选圆
        :return: True 圆心和角点之间的距离不合理
        """

        if self._isinQurd(a, b, c, d, (keypoint.pt[0], keypoint.pt[1])):
            return True
        l = [a, b, c, d]
        # print(keypoint)
        # print("edgel", edgel)
        min_num=0
        min_len=9999
        for i in range(len(l)):
            # print("len",self._getlength(p, keypoint.pt))
            tmp_min=self._getlength(l[i], keypoint.pt)
            if min_len>tmp_min:
                min_len=tmp_min
                min_num=i
        max_num=(min_num+2)%4
        xl_0 = np.array(l[min_num] - l[max_num])
        xl_1=np.array(keypoint.pt-l[max_num])
        res=self.getcos(xl_0,xl_1)
        print("res:",res)

        if 1-res<0.01:
            return False
        return True

    def _circle_nearby(self,original_image, a, b, c, d):
        '''
        :a,b,c,d:apriltag四个角点
        :return :image and circle points
        '''
        '''四边形边长'''
        edgel = (self._getlength(a, b) + self._getlength(c, b) + self._getlength(c, d) + self._getlength(a, d)) / 4

        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.01
        # params.filterByInertia=True
        # params.minInertiaRatio=0.1
        # params.maxInertiaRatio=1
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(original_image)
        kps = keypoints.copy()
        for keypoint in kps:
            if self._regeonillegal(a, b, c, d, edgel, keypoint):

                keypoints.remove(keypoint)
        return keypoints, kps

    def _sort(self,qlist,aprillist):
        """根据正方形角点对圆点进行排序，确保圆点的编号和离他最近的正方形的编码是一样的"""
        return_list=[1,2,3,4]
        for q in qlist:
            minl,minid = 100000,5
            for i in range(len(aprillist)):
                tl=self._getlength(q,aprillist[i])
                if minl>tl:
                    minl=tl
                    minid=i
            return_list[minid]=q
        return return_list

    def _tags_2points(self,tagps,shape):
        '''
        :param tagps:图像中标签的五个角点的坐标信息
        :return: 返回能够包含各个标签的最小的正方形的边界信息
        '''
        maxpixhang = 0
        maxpixlie = 0

        (hang,lie) = shape
        minpixhang = hang
        minpixlie = lie
        for points in tagps:
            maxpixhang = max(maxpixhang, points[1])
            maxpixlie = max(maxpixlie, points[0])
            minpixhang = min(minpixhang, points[1])
            minpixlie = min(minpixlie, points[0])
        rangeh=(maxpixhang-minpixhang)//2
        rangel=(maxpixhang-minpixhang)//2
        if minpixlie != lie:
            list = [[max((int)(minpixhang - rangeh), 0), max((int)(minpixlie - rangeh), 0)],
                    [min((int)(maxpixhang + rangel), hang), min((int)(maxpixlie + rangel), lie)]]
        else:
            list = [[maxpixhang, maxpixlie], [minpixhang, minpixlie]]
        return list

    def getcos(self,a, b):
        la = np.sqrt(a.dot(a))
        lb = np.sqrt(b.dot(b))
        return a.dot(b) / (la * lb)
    def detect(self,img):
        return_info=[]
        gray = img.copy()
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        apriltag_detections=self._detector.detect(gray)
        canny = cv2.Canny(gray, 0, 255)  # cv2.dilate(canny,(3,3))
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 检测所有的轮廓  外层是顶层
        edges = np.array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 0]])
        showpic = img.copy()
        '''检测方形'''
        allsh=showpic.copy()
        for det in apriltag_detections:

            subarea=self._tags_2points(det.corners,gray.shape)
            ROI = subarea
            gray_roi = gray[ROI[0][0]:ROI[1][0] + 1, ROI[0][1]:ROI[1][1] + 1]
            kernal=np.ones((3,3),np.uint8)
            gray_roi=cv2.erode(gray_roi,kernal)

            startarr = np.array([ROI[0][1], ROI[0][0]])
            det.center -= startarr
            for i in range(4):
                det.corners[i] -= startarr
            rpoints=det.corners
            points = det.corners.astype('int')
            '''对圆形检测区域进行设置，按照边长进行适当的扩展'''
            kps, allkps = self._circle_nearby(gray_roi, rpoints[0], rpoints[1], rpoints[2], rpoints[3])
            # print(kps)
            # print(allkps)
            # xl_0 = np.array(rpoints[1] - rpoints[3])
            # print(xl_0)
            # xl_1=np.array(rpoints[0]-rpoints[2])
            # print(xl_1)
            # for i in range(5):
            #     print(i)
            #     for j in range(4):
            #         xl_2=np.array(allkps[i].pt-rpoints[j])
            #
            #         print(self.getcos(xl_1,xl_2))
            #         print(self.getcos(xl_0, xl_2))
            kplist = []
            for kp in kps:
                kplist.append((kp.pt[0], kp.pt[1]))
            if self._debug:
                for j in range(4):
                    cv2.line(showpic, tuple(points[edges[j, 0]]), tuple(points[edges[j, 1]]), (0, 0, 255), 2)
                showpic = cv2.drawKeypoints(showpic, kps, showpic, (0, 0, 255),
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                allsh = cv2.drawKeypoints(allsh, allkps, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("imgall", allsh)
                cv2.imshow("img", showpic)
                cv2.waitKey()
            # cv2.imshow("origin",gray_roi)
            # cv2.waitKey()
            if len(kplist)!=4:
                continue
            kplist=self._sort(kplist,rpoints)
            kplist=np.asarray(kplist)
            startarr = np.array([ROI[0][1], ROI[0][0]])
            det.center += startarr
            # print("kps",kps[0].pt)
            for i in range(4):
                det.corners[i] += startarr
                kplist[i]+=startarr
                kps[i].pt=tuple(np.array(kps[i].pt)+startarr)

            detetion=Detection()
            detetion.tag_family=det.tag_family
            detetion.apcorners=det.corners
            detetion.tag_id=det.tag_id
            detetion.center=det.center
            detetion.corners=kplist
            detetion.keypoints=kps
            return_info.append(detetion)

        return return_info #,showpic


if __name__ == '__main__':
    filepath="nt2.bmp"
    img=cv2.imread(filename=filepath)


    at_detector=Detector(families='tag36h11',debug=True)
    detections=at_detector.detect(img)
    print(detections)
    cv2.imshow("origin image",img)
    # cv2.imshow("output image",resultimage)
    cv2.waitKey()
