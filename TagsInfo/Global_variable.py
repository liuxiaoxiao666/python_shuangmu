import numpy as np
import threading
import pupil_apriltags as apriltag
exitFlag = False

tagnum=20
# img1 = np.zeros((2048, 2592, 3))
# img1 = img1.astype(np.uint8)
# img2 = np.zeros((2048, 2592, 3))
# img2 = img2.astype(np.uint8)
img1 = np.zeros((3000, 4096))
img1 = img1.astype(np.uint8)
img2 = np.zeros((3000, 4096))
img2 = img2.astype(np.uint8)
img_list={
    'img1':img1,
    'img2':img2
}
idDict={}
event3 = threading.Event()
event3.set()
result = []
shared_lock = threading.Lock()
count = [0 for i in range(tagnum)]
ave = [0 for i in range(tagnum)]
avepoints=[[0,0,0,0] for i in range(tagnum)]
depthlist = [[] for i in range(tagnum)]
depthlistpoints=[[[],[],[],[]] for i in range(tagnum)]
jittycnt = [0 for i in range(tagnum)]
subPointsListL = [[] for i in range(tagnum)]
aveSubPointsListL = [[] for i in range(tagnum)]
subPointsListR = [[] for i in range(tagnum)]
aveSubPointsListR = [[] for i in range(tagnum)]
#print(subPointsListL)
subPointsCntL = [0 for i in range(tagnum)]
subPointsCntR = [0 for i in range(tagnum)]
imgFlagL = 0
imgFlagR = 0
leftCameraJittyList = [0 for i in range(tagnum)]
rightCameraJittyList = [0 for i in range(tagnum)]
subaveleft = [0 for i in range(tagnum)]
subaveright = [0 for j in range(tagnum)]
#print(subaveleft)
at_detector = apriltag.Detector(families='tag36h11')
resdata=[]

rescnt = 0
res=[]
tagpos = {
    'tags': None,
    'END_TIME': None,
    'fps':0
}

testangledist={
'flagangle':False,
'flagangle2':False,
'xnow':0,
'ynow':0,
'znow':0
}

testmovdist={
'flagmov':False,
'flagmov2':False,
'xnow':0,
'ynow':0,
'znow':0,
'save':False,
'wd_num':0,
'wd_fname':'',
'wd_cnt':0,
'wd_v':[],
'init_num':0,
'xsum':0,
'ysum':0,
'zsum':0,
}

'''用于区域的提取保存'''
initnum=0  #初始化状态，用于确定标签的总体数量，取前五帧的最大值
detect_status=True  #检索状态 True：图片全局检索状态，False：标签位置跟踪状态
tagsnum=0  #区域中总的标签的数量
tagsinfo=[]  #保存区域中这些标签的id和角点位置信息

'''用于区域的提取保存'''
roi_p={
    "initstart":True,  #初始化开始，由界面按钮控制
    "initnum":0,  #初始化状态，用于确定标签的总体数量，取前五帧的最大值
    "roi_state":False,  #检索状态 False：图片全局检索状态，True：标签位置跟踪状态
    "tagsnum":0,  #区域中总的标签的数量
    "tagsinfol":[],  #保存左相机区域中这些标签的id和角点位置信息
    "tagsinfor":[],  #保存右相机区域中这些标签的id和角点位置信息
    "all_view":True #全局检索
}
'''保存数据相关变量'''
save_p={
    'catch_v':[],
    'catch_num':0,
    'catch_flag':False,
    'catch_fname':'',
    'finish':False
}
test_area={
    'ta_v':[],
    'ta_num':0,
    'ta_flag':False,
    'ta_fname':'',
    'finish':False
}
tc = {
    'left': 0,
    'right': 0,
    'caltime': 0
}
ids={
    'tl':[],
    'tr':[]
}
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
width=4096
height=3000
B=0.2095898872560685789837477057394
