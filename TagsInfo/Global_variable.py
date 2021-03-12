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
'znow':0
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
tc = {
    'left': 0,
    'right': 0,
    'caltime': 0
}
ids={
    'tl':[],
    'tr':[]
}
camera_matrix1= np.array([[3574.09836290701, 0.919498022419275, 2031.76962647928],
                             [0.00000000e+00, 3572.77912032506, 1526.47427445566],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_matrix2= np.array([[3575.20789111229 ,-0.463657413728510 ,2062.40502075186],
                            [0, 3575.06428531935, 1542.00719185697],
                            [0.00000000e+00 ,0.00000000e+00 ,1]])

dist_coeffs1= np.array([-0.099178476901251 , 0.138102681914255, 6.274110222273973e-04,-2.724380259422465e-04 ,-0.106909615003580])
dist_coeffs2= np.array([-0.099518437619588 ,  0.157127467456666, 0.001132447921996,  1.331016115699895e-04 ,-0.268492755010379])
R= np.array([[ 0.985210517889707 ,-0.006972366641296 , 0.171206371210053],
                 [ 0.006583654485686 , 0.999974299994116 , 0.002838105863397],
                [-0.171221759519952,  -0.001668968153708,  0.985231101626514]])
T= np.array([-0.2090552072686908, -0.198426078216100e-03, 0.018326765876410498 ])
width=4096
height=3000
B=2.0985706899352366388199261083491e-1
