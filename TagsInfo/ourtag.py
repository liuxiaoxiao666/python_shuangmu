import cv2
import numpy as np
import time
import math
from skimage.morphology import convex_hull_image,convex_hull,convex_hull_object

# image_size=()
# 得到两点之间的距离
def distance(point1, point2):
    return (point1[0]-point2[0])**2+(point1[1]-point2[1])**2

#加入编码3*3，汉明距离为3，总共16个标签t
tag_dic9h3=[0x1a6, 0x16b, 0xf5, 0xba, 0x1ce,0x158, 0x11d, 0x145,
        0x1a8, 0x6e, 0x96, 0x5b, 0x134, 0x1d2, 0x171, 0x1fc]

class decoder:
    def __init__(self, image, image_size=(1024,1024), tag_family=tag_dic9h3, COEFFICIENT=0.02):
        self.image = image
        self.datas = []
        self.image_size = image_size
        self.tag_family = tag_family
        self.COEFFICIENT=COEFFICIENT

    #输入是五边形从左上角开始顺时针旋转的五个点，这个函数把顺序调整为短一，短二。。顺时针旋转
    def sort(self, cnt):
        # print("before",cnt)
        lengths = [distance(cnt[i][0],cnt[i+1][0]) for i in range(4)]
        lengths.append(distance(cnt[4][0],cnt[0][0]))
        min_index = np.argmin(lengths)
        cnt_copy = cnt.copy()
        for i in range(5):
            count = min_index+i
            count = count if count < 5 else count - 5
            cnt[i] = cnt_copy[count]
        # print("after:",cnt)
        return

    # 输入sort好的顶点坐标，得到9个数据中心点坐标
    # 输出点坐标顺序为，当最短边在左上角时，按照先从左往右，后从上到下的顺序
    def getNineCenter(self, sorted_cnt):
        # print(sorted_cnt)
        tile_d1 = np.array([int((sorted_cnt[2][0][i] - sorted_cnt[3][0][i])/5) for i in range(2)])
        tile_d2 = np.array([int((sorted_cnt[3][0][i] - sorted_cnt[4][0][i])/5) for i in range(2)])
        center =  np.array([min(max(int((sorted_cnt[2][0][i] + sorted_cnt[4][0][i])/2),0),self.image_size[1-i]) for i in range(2)])
        centers = np.array([[center-tile_d1*i+tile_d2*j for i in range(-1,2) for j in range(-1,2)]])
        # print("sorted_cnt",sorted_cnt,"tile_d1",tile_d1,"tile_d2",tile_d2,"center",center)
        return centers

    #输入9个中心点坐标和图像，按照黑色为1，白色为0的方式输出最终编码
    def getData(self, centers, image):
        data = ''
        for point in centers[0]:
            if image[point[1]][point[0]] > 127:         #point的xy坐标顺序和image的坐标顺序是反过来的*__*
                data+='0'
            else:
                data+='1'
        return data

    #判断是否是正常的顺时针旋转
    def isConvex(self, cnt):
        size=len(cnt)
        for i in range(size-1):
            a=np.array(cnt[(i)%size])[0]
            b=np.array(cnt[(i+1)%size])[0]
            c=np.array(cnt[(i+2)%size])[0]
            a[1]=-a[1]
            b[1]=-b[1]
            c[1]=-c[1]
            if (a[0]-c[0])*(b[1]-c[1])-(b[0]-c[0])*(a[1]-c[1])>0:
                return False
        return True


    def decode(self):
        #转为灰度值格式
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #转为黑白二值格式
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        #找到所有的轮廓线
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #找到所有的五边形
        potential_pentagons =[]
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            # approx=cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(cnt, self.COEFFICIENT * peri, True)
            if cv2.contourArea(approx)<100:
                continue
            if len(approx) == 5 and self.isConvex(approx):
                potential_pentagons.append(approx)
                cv2.drawContours(self.image, [approx], -1, (255, 0, 100), 3)

        for cnt in potential_pentagons:
            self.sort(cnt)
            centers = self.getNineCenter(cnt)
            data = self.getData(centers, thresh)
            # print(data)
            if int(data,2) in self.tag_family:
                print("tag is ",  self.tag_family.index(int(data,2)))
                self.datas.append(self.tag_family.index(int(data,2)))
        return self.datas

if __name__ == '__main__':
    COEFFICIENT=0.02
    # path = "tag.png"
    # image = cv2.imread(path)


    cap = cv2.VideoCapture(0)
    # cap.set(3, 1920)
    # cap.set(4, 1080)
    fps = 24
    window = 'Camera'
    cv2.namedWindow(window)
    while cap.grab():
        success, frame = cap.retrieve()
        if not success:
            break
        image=frame
        image_size=frame.shape
        # print(image.shape)

        decoder1 = decoder(image, image_size)
        tags = decoder1.decode()

        cv2.imshow(window, image)
        k = cv2.waitKey(1000 // int(fps))

        if k == 27:
            break
    cap.release()
