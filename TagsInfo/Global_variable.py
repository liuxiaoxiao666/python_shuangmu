import numpy as np
import threading
import pupil_apriltags as apriltag
# import ac_tag as apriltag
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
# camera_matrix1 = np.array([[3.574039860680742e+03, 1.098719182713949, 2.034292584522333e+03],
#                            [0.00000000e+00, 3.572516961910460e+03, 1.506600908172405e+03],
#                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# camera_matrix2 = np.array([[3.577313313259017e+03, -1.129222862508354, 2.055469746090763e+03],
#                            [0, 3.574454683953720e+03, 1.525514668948427e+03],
#                            [0.00000000e+00, 0.00000000e+00, 1]])
#
# dist_coeffs1 = np.array(
#     [-0.092992078636370, 0.096222535649721, 4.719115037495438e-05, -2.361695408341265e-05, -0.008870832773170])
# dist_coeffs2 = np.array(
#     [-0.094914780676932, 0.096928513695695, 3.747002421475770e-04, 1.843273274888114e-05,-0.008310968949008])
# R = np.array([[0.985055260835653,-0.005851365777969, 0.172139172237281],
#               [0.005512578564103,0.999981813938738, 0.002446072232485],
#               [-0.172150354567101,-0.001460585610075, 0.985069602673891]])
# T = np.array([-0.2087596722021999, -0.430732180863642e-03, 0.018631547757230287])
# width = 4096
# height = 3000
# B=0.2095898872560685789837477057394

# camera_matrix1 = np.array([[3.564100914701934e+03, -5.354055265107811, 2.018475682613483e+03],
#                            [0.00000000e+00, 3.562907129221029e+03, 1.517493515200955e+03],
#                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# camera_matrix2 = np.array([[3.535439675347217e+03, -3.452783153381272, 2.051321308630927e+03],
#                            [0, 3.532061057382973e+03, 1.539273250373704e+03],
#                            [0.00000000e+00, 0.00000000e+00, 1]])
#
# dist_coeffs1 = np.array(
#     [-0.098549396542964, 0.119462724440395, 2.608416883084959e-04, 1.265176442259160e-04, -0.053361998496539])
# dist_coeffs2 = np.array(
#     [-0.088506610293866,0.044832527198377, -4.573903914640852e-05, -3.508585444409916e-05,0.080072156254630])
# R = np.array([[0.985321244897772,-0.006303705895017, 0.170593984785791],
#               [0.006274658954030,0.999980062479331, 7.094355818489540e-04],
#               [-0.170595055637953,3.713971234576565e-04, 0.985341153639723]])
# T = np.array([-0.2097780450660247, 0.347813027543609e-03, 0.013141942047608467])
# width = 4096
# height = 3000

# B=0.21036229127314401357310955673791

camera_matrix1 = np.array([[3.574619125510544e+03, 0.744853664962436, 2.032757680687241e+03],
                           [0.00000000e+00,3.572146285136761e+03, 1.510302589708326e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_matrix2 = np.array([[3.574857374721066e+03, 0.277431611699156, 2.056487900434615e+03],
                           [0, 3.571938956634337e+03, 1.530727803797534e+03],
                           [0.00000000e+00, 0.00000000e+00, 1]])

dist_coeffs1 = np.array(
    [-0.093277415458552, 0.095278921484504, 2.526471111957186e-04, -5.391700239658532e-05, -0.008051622802359])
dist_coeffs2 = np.array(
    [-0.094449256937348,0.094682817398000, 6.598987620566508e-04, -3.934049076258134e-05,-0.007851681324673])
R = np.array([[0.985059685881259,-0.006093413615824,0.172105448960867],
              [0.005774858025856,0.999980062479331, 0.002351554421272],
              [-0.172116432295979,-0.001322536926325, 0.985075725327643]])
T = np.array([-0.2091983740647370, -0.559064086847281e-03, 0.018018816071774392])
width = 4096
height = 3000
B=0.20997368881984062479738258903143
