from queue import Queue
import cv2
import pupil_apriltags as apriltag
from pupilApriltags.tools import *
import numpy as np
import pyrealsense2 as rs
import time
from Kalman import *

fps = 1000
TAG_SIZE = 0.020
avemap = [[-4, -2], [-4, -1], [-4, 0], [-4, 1], [-4, 2],
          [-3, -3], [-3, -2], [-3, -1], [-3, 0], [-3, 1], [-3, 2], [-3, 3],
          [-2, -4], [-2, -3], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-2, 3], [-2, 4],
          [-1, -4], [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4],
          [0, -4], [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
          [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
          [2, -4], [2, -3], [2, -2], [2, -1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
          [3, -3], [3, -2], [3, -1], [3, 0], [3, 1], [3, 2], [3, 3],
          [4, -2], [4, -1], [4, 0], [4, 1], [4, 2]]


class SortUnit:
    def __init__(self, dis, num):
        self.dis=dis
        self.num = num


def getdis(sortunit):
    return sortunit.dis


class Plane:
    def __init__(self, xs, ys, zs):
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.pointnum = xs.shape[0]

    def calculate(self):
        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        for i in range(self.pointnum):
            A[0, 0] = A[0, 0] + self.xs[i] * 2
            A[0, 1] = A[0, 1] + self.xs[i] * self.ys[i]
            A[0, 2] = A[0, 2] + self.xs[i]
            A[1, 1] = A[1, 1] + self.ys[i] ** 2
            A[1, 2] = A[1, 2] + self.ys[i]
            B[0, 0] = B[0, 0] + self.xs[i] + self.zs[i]
            B[1, 0] = B[1, 0] + self.ys[i] + self.zs[i]
            B[2, 0] = B[2, 0] + self.zs[i]

        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = self.pointnum

        A_inv = np.linalg.inv(A)
        p = np.dot(A_inv, B)
        return p


class Movemean:
    def __init__(self,alpha):
        self.alpha=alpha
        self.queue=Queue(maxsize=5)
        self.sum=0
        self.weight=alpha+alpha**2+alpha**3+alpha**4+alpha**5
    def average(self,value):
        if(self.queue.full()):
            self.sum=(self.sum-self.queue.get()*(self.alpha**5)+value)*self.alpha
            self.queue.put(value)
        else:
            self.queue.put(value)
            self.sum=(self.sum+value)*self.alpha
        return self.sum/self.weight



def main():
    filecounter=0
    start=time.time()
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_device('841612070330')
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()

    # 对齐
    align_to = rs.stream.color
    align = rs.align(align_to)

    print(intr)  # 获取内参 width: 640, height: 480, ppx: 319.115, ppy: 234.382, fx: 597.267, fy: 597.267, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]
    print(intr.ppx)  # 获取指定某个内参
    # cap = cv2.VideoCapture(0)
    # cap.set(3,1920)
    # cap.set(4,1080)
    window = 'Camera'
    cv2.namedWindow(window)
    at_detector = apriltag.Detector(families='tag36h11')  # for windows

    last_distance = 0
    distance_map = []
    error_value_map = []
    error_pct_map = []

    km_filter = KalmanFilter(0.001, 0.1)  # define a kalman filter
    moveaverage=Movemean(1)
    filename="D:\santan\data\data"+str(filecounter)+".txt"
    f = open(filename, 'w')
    datalist = []
    datamean=[]
    alist = []
    planelist = []
    planekalmanlist = []
    selectpdis=[]
    seldkm=[]
    counter = 0
    datalist.append([])
    datamean.append([])
    alist.append([])
    planelist.append([])
    planekalmanlist.append([])
    selectpdis.append([])
    seldkm.append([])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    px=[]
    py=[]
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)  # 对齐
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            dpt_frame = depth_frame.as_depth_frame()

            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # success, frame = cap.retrieve()
            # if not success:
            #     break
            frame = color_image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = at_detector.detect(gray, estimate_tag_pose=True,
                                            camera_params=[intr.fx, intr.fy, intr.ppx, intr.ppy], tag_size=TAG_SIZE)
            show = None
            if (len(detections) == 0):
                show = frame
            else:
                # print("%d apriltags have been detected." % len(detections))
                show = frame
                edges = np.array([[0, 1],
                                  [1, 2],
                                  [2, 3],
                                  [3, 0]])

                for tag in detections:

                    translation = tag.pose_t
                    # 3*3
                    rotation = tag.pose_R
                    F = np.array([1, 0, 0,
                                  0, -1, 0,
                                  0, 0, 1]).reshape((3, 3))

                    # fixed_rot = F * rotation
                    fixed_rot = np.dot(F, rotation)
                    yaw, pitch, roll = wRo_to_euler(fixed_rot)

                    # print("translation:",translation," rotation:",rotation," F:",F," fixed_rot:",fixed_rot)
                    point = tag.corners.astype('int')
                    print(point[0])
                    testcorners=point.astype(np.float32)

                    cv2.cornerSubPix(gray,testcorners,(11,11),(-1, -1), criteria)
                    print(testcorners[0])
                    for j in range(4):
                        cv2.line(show, tuple(point[edges[j, 0]]), tuple(point[edges[j, 1]]), (0, 0, 255), 2)

                    distance = np.linalg.norm(translation)    # the distance
                    distance_map.append(distance)
                    if last_distance is not 0:
                        error_value = abs(distance - last_distance)
                        error_value_map.append(error_value)
                        error_percent = abs(distance - last_distance) / last_distance * 100
                        error_pct_map.append(error_percent)
                        # print(f'error value:{round(error_value, 6)}')
                        # print('error percent:''%.4f%%' % error_percent)
                        # print(f'error value mean:{round(np.mean(error_value_map), 6)}')
                        # print(f'error percent mean:{round(np.mean(error_pct_map), 4)}%')
                        # print(f'std:{np.std(distance_map)}')
                    last_distance = distance
                    # print('distance={}m x={} y={} z={} yaw={} pitch={} roll={}'
                    #       .format(distance, translation[0], translation[1], translation[2],
                    #               yaw, pitch, roll))
                    '''region average'''

                    xs = []
                    ys = []
                    zs = []
                    # point=[[624,392],]
                    for temp in avemap:
                        xs.append(point[0][0] + temp[0])
                        ys.append(point[0][1] + temp[1])
                        zs.append(dpt_frame.get_distance(point[0][0] + temp[0], point[0][1] + temp[1]))
                    pln = Plane(np.array(xs), np.array(ys), np.array(zs))
                    P = pln.calculate()
                    # print(P)
                    # P保留了平面参数
                    planedepth = P[0][0] * point[0][0] + P[1][0] * point[0][1] + P[2][0]

                    # print("simple plane depth:",planedepth)
                    '''delete the points that were too far away from the plane'''
                    distoplane=[]
                    for temp in avemap:
                        distoplane.append(SortUnit(
                            (P[0][0] * (point[0][0] + temp[0])+ P[1][0] * (point[0][1]+temp[1] )+ P[2][0]-dpt_frame.get_distance(point[0][0] + temp[0], point[0][1] + temp[1])),
                            avemap.index(temp))
                        )
                    distoplane.sort(key=getdis)
                    # selectdis=distoplane[23:46]
                    selectdis=distoplane[10:59]
                    xs.clear()
                    ys.clear()
                    zs.clear()
                    for temp in selectdis:
                        xs.append(point[0][0] + avemap[temp.num][0])
                        ys.append(point[0][1] + avemap[temp.num][1])
                        zs.append(dpt_frame.get_distance(point[0][0] + avemap[temp.num][0], point[0][1] + avemap[temp.num][1]))

                    plnsel = Plane(np.array(xs), np.array(ys), np.array(zs))
                    Pn = plnsel.calculate()
                    planedepthsel = (Pn[0][0] * point[0][0] + Pn[1][0] * point[0][1] + Pn[2][0])
                    ''''''

                    '''直接获取realsense距离'''
                    depth_dis = dpt_frame.get_distance(*list(point[0]))
                    fivebolckave=moveaverage.average(depth_dis)

                    if depth_dis != 0:
                        kmdis = km_filter.kalman(depth_dis)
                        # plkmdis = planekl_filter.kalman(planedepth)
                        # seldiskm=sel_filter.kalman(planedepthsel)
                        kmd=km_filter.kalman(fivebolckave)
                        # print(f'depth_dis\t={depth_dis * 1000}mm')  # realsense输出距离
                        text="R:"+str(depth_dis*1000)
                        cv2.putText(show,text,(40,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
                        text="A:"+str(fivebolckave*1000)
                        cv2.putText(show,text,(40,70),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
                        text="K:"+str(kmd*1000)
                        cv2.putText(show,text,(40,90),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
                        # print(f'kmdis\t\t={kmdis * 1000}mm')  # realsense输出距离
                        # print("plane point depth after select 40% points:", planedepthsel)

                    if 100 * len(detections) <= counter < 200 * len(detections):
                        datalist[detections.index(tag)].append(depth_dis)
                        datamean[detections.index(tag)].append(fivebolckave)
                        alist[detections.index(tag)].append(kmd)
                        # print(f'depth_dis={depth_dis * 1000}mm')  # realsense输出距离
                        # print("kalman depth dis:", kmdis * 1000, 'mm')
                        # print("plane point depth after select 40% points:", planedepthsel)

                        counter += 1
                        print('counter num',counter)
                    elif counter == 200 * len(detections) :

                        # print("time cost:",time.time()-start)
                        l = len(datalist)
                        for i in range(l):
                            ll = len(datalist[i])
                            # for disdata in l:
                            for j in range(ll):
                                f.writelines(str(datalist[i][j] * 1000) +
                                             '\t' + str(datamean[i][j] * 1000) +
                                             '\t' + str(alist[i][j] * 1000) +
                                             # '\t' + str(planelist[i][j]* 1000) +
                                             # '\t' + str(planekalmanlist[i][j]* 1000) +
                                             # '\t' + str(selectpdis[i][j]* 1000) +
                                             # '\t' + str(seldkm[i][j]* 1000)+
                                             '\n')
                        counter+=1
                        datalist[0].clear()
                        datamean[0].clear()
                        alist[0].clear()
                        f.close()
                        filename = "D:\santan\data\data" + str(1+filecounter) + ".txt"
                        filecounter=filecounter+1
                        f = open(filename, 'w')

                    else:
                        if counter<200:
                            counter = counter + 1
                        else:
                            counter = 0
                            pass

            ########################
            num_detections = len(detections)
            cv2.imshow(window, show)
            k = cv2.waitKey(1000 // int(fps))
            if k == 27:
                break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
