from TagsInfo.getTags import *
from TagsInfo.tools import *
from TagsInfo.Global_variable import *
tans=0
def getimg():
    global img_list
    # print(img1)
    pic1=img_list['img1'].copy()
    pic2=img_list['img2'].copy()
    return pic1,pic2

def calculateThread():

    global P1, P2,R1,R2, map1x, map1y, map2x, map2y,camera_matrix1,camera_matrix2,dist_coeffs1,dist_coeffs2,R,T,tagpos
    P1, P2,R1,R2, map1x, map1y, map2x, map2y = rectify()
    camera_matrix1 = np.array(camera_matrix1)
    camera_matrix2 = np.array(camera_matrix2)
    dist_coeffs1 = np.array(dist_coeffs1)
    dist_coeffs2 = np.array(dist_coeffs2)
    R = np.array(R)
    T = np.array(T)
    while True:

        timett = time.time()
        global  depthlist
        # print("aaa:",time.time()-timett)
        shared_lock.acquire()
        pic1,pic2=getimg()
        # map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, (width, height), cv2.CV_32FC1)
        # map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, (width, height), cv2.CV_32FC1)
        # # 畸变校正和立体校正
        # rectifyed_img1 = cv2.remap(pic1, map1x, map1y, cv2.INTER_AREA)
        # rectifyed_img2 = cv2.remap(pic2, map2x, map2y, cv2.INTER_AREA)
        # cv2.imwrite("img1.bmp", rectifyed_img1)
        # cv2.imwrite("img2.bmp", rectifyed_img2)

        shared_lock.release()
        # print("bbb:",time.time()-timett)
        global result
        res = getTags(pic1, pic2,P1, P2,R1,R2)
        '''
        返回值结构如下：tags points:
        [[ 0.11659958 -0.05395726  0.76397341]
         [ 0.13686914 -0.05381902  0.76395789]
         [ 0.13728232 -0.07366932  0.76380179]
         [ 0.1172662  -0.07382156  0.76381767]]
        disToSurface:
        764.4685496605849
        rotation:
        [0.393231018631025, 89.60922425678058, 90.04387360061794]
        '''
        # print(res[1][0])
        # if res[0] == 1:
        #     for i in range(len(res[1])):
        #         if (res[1][i].points[0][2]*1000 != 0):
        #             global rescnt
        #             if rescnt < 10:
        #                 rescnt = rescnt + 1
        #             else:
        #                 result.append(res[1][i].points[0][2]*1000)
                        # print(result[-1],"tagid:",res[2][i])
        #showTarget(result)
        fpstime=time.time()-timett
        # print(fpstime)
        tagpos['tags']=res
        # print(res[1])
        '''保存数据相关'''
        global save_p
        # print(res)
        if res != None and type(res[1]) == list:
            if type(res[1][0]) == Tag:
                if save_p['catch_flag'] and len(save_p['catch_v']) < save_p['catch_num']:
                    print(len(save_p['catch_v']))

                    save_p['catch_v'].append(res[1])
                elif save_p['catch_flag'] and len(save_p['catch_v']) == save_p['catch_num']:
                    with open(save_p['catch_fname'], 'w', newline='') as f:
                        mywrite = csv.writer(f)
                        da=[]
                        for i in range(len(save_p['catch_v'][0])):
                            da =da+list(['tag '+str(i)+' x','tag '+str(i)+' y','tag '+str(i)+' z',
                                            'tag '+str(i)+' rx','tag '+str(i)+' ry','tag '+str(i)+' rz'])
                        mywrite.writerow(da)
                        for tags in save_p['catch_v']:
                            data = []
                            for tag in tags:
                                temtaag=tag.points[0]*1000
                                data += temtaag.tolist()

                                data += tag.rotation
                            mywrite.writerow(data)
                    print("output")
                    save_p['catch_v'].clear()
                    # b.config(state=tk.ACTIVE)
                    save_p['catch_flag'] = False
                    save_p['finish']=True

