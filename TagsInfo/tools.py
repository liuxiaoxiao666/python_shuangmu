from TagsInfo.Global_variable import *
import numpy as np
from numpy import mean, absolute
import cv2
import ctypes
import inspect
import matplotlib.pyplot as plt
import csv
import math

def getcos(a,b):
    la=np.sqrt(a.dot(a))
    lb=np.sqrt(b.dot(b))
    return a.dot(b)/(la*lb)

def standardRad(t):
    TWOPI = 2.0 * math.pi
    if t >= 0.:
        t = math.fmod(t + math.pi, TWOPI) - math.pi
    else:
        t = math.fmod(t - math.pi, -TWOPI) + math.pi
    return t


def wRo_to_euler(wRo):
    yaw = standardRad(math.atan2(wRo[1, 0], wRo[0, 0]))
    c = math.cos(yaw)
    s = math.sin(yaw)
    pitch = standardRad(math.atan2(-wRo[2, 0], wRo[0, 0] * c + wRo[1, 0] * s))/math.pi*180
    roll = standardRad(math.atan2(wRo[0, 2] * s - wRo[1, 2] * c, -wRo[0, 1] * s + wRo[1, 1] * c))/math.pi*180
    # yaw,pitch,roll分别是绕z，y，x轴旋转的欧拉角

    # print(yaw/math.pi*180, pitch, roll)
    '''测试发现x和z需要倒一下'''
    # return [yaw/math.pi*180, pitch, roll]
    return [roll,pitch,yaw/math.pi*180 ]




def getangle(Rx,Ry,Rz):
    '''
    Rx:标签x轴向量，下同
    '''
    cx = np.array([1, 0, 0])
    cy = np.array([0, 1, 0])
    cz = np.array([0, 0, 1])
    h=np.array([[getcos(cx,Rx),getcos(cx,Ry),getcos(cx,Rz)],
            [getcos(cy,Rx),getcos(cy,Ry),getcos(cy,Rz)],
            [getcos(cz,Rx),getcos(cz,Ry),getcos(cz,Rz)]])
    return wRo_to_euler(h)

def mad(data):
    return mean(absolute(data - mean(data)))


def showTarget(list):
    if len(list) == 0:
        return
    var = np.var(list)  # 方差
    std = np.std(list)  # 标准差
    diff = max(list) - min(list)  # 最值差
    Mad = mad(list)  # 平均绝对偏差
    print("方差：%f" % var)
    print("标准差：%f" % std)
    print("最值差：%f" % diff)
    print("平均绝对偏差：%f\n" % Mad)


def getDis(p1, p2):
    temp = 0
    for i in range(len(p1)):
        temp += (p1[i] - p2[i]) * (p1[i] - p2[i])
    return np.sqrt(temp)


def rectify():
#50-66
# camera_matrix1=[[2.594551490224181e+03,1.414493920473977,1.277866230941417e+03],[0,2.595482721834991e+03,1.023794628734023e+03],[0,0,1]]
# camera_matrix2=[[2.609813133521772e+03,0.257631167606638,1.286434925362545e+03],[0,2.610271153947829e+03,1.013280115878406e+03],[0,0,1]]
# dist_coeffs1=[-0.100348960344808,0.133716561969318,-8.493160863680077e-04,8.299437754832901e-05,0.082265332548897]
# dist_coeffs2=[-0.101388812897774,0.107162714687271,-3.288372572244118e-04,6.127374209211794e-04,0.215647646870015]
# R=[[0.937291667043855,-0.003327046465135,0.348530144538418],[0.003864303018429,0.999992175444646,-8.462928967608300e-04],[-0.348524601789219,0.002140049369566,0.937297189869020]]
# T=[-4.275084535649460e-01,2.76082551765435e-04,8.2287680478249330e-02]

#82-100
# camera_matrix1=[[2.587227409939453e+03,0.621536830532253,1.281452076259251e+03],[0,2.587399760650868e+03,1.026691691847859e+03],[0,0,1]],
# camera_matrix2=[[2.608643607182612e+03,-0.519194868178063,1.284531384380881e+03],[0,2.608794600682819e+03,1.010630878614789e+03],[0,0,1]],
# dist_coeffs1=[-0.108442050624486,0.171858179826349,-2.509338450222089e-04,-1.810429201295466e-04,0.052919632121287],
# dist_coeffs2=[-0.105239417737543,0.174158006091541,-3.136073313482666e-04,8.271496597030550e-04,0.087934209622312],
# R=[[0.937165564422082,-0.004069578501697,0.348861209354464],[0.003848962945429,0.999991714180549,0.001325539321404],[-0.348863713139796,1.005040615957793e-04,0.937173462894276]],
# T=[-4.253612693960524e-01,3.60750275113523e-04,8.3948013017829880e-02]

#all
# camera_matrix1=[[2.590573926058853e+03,0.570668242395603,1.281386281646616e+03],[0,2.590838525076922e+03,1.026087187844178e+03],[0,0,1]],
# camera_matrix2=[[2.610837096823258e+03,-0.184668074761606,1.282810140389639e+03],[0,2.611213972276432e+03,1.011861362584955e+03],[0,0,1]],
# dist_coeffs1=[-0.108732897469549,0.185354970950092,-3.407961259096508e-04,-1.725512221026316e-04,-0.011769941784738],
# dist_coeffs2=[-0.100416206759202,0.121179439570604,-2.523490854052689e-04,6.660819372768040e-04,0.245661481325296],
# R=[[0.936850063838506,-0.004073039893709,0.349707546718413],[0.004010721907871,0.999991549896792,9.023551354125222e-04],[-0.349708266982015,5.572082527550326e-04,0.936858483189109]],
# T=[-4.259527780672074e-01,7.2745266768265e-04,8.4082092801290640e-02]

    '''
    camera_matrix1 = np.array([[3574.09836290701, 0.919498022419275, 2031.76962647928],
                               [0.00000000e+00, 3572.77912032506, 1526.47427445566],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    camera_matrix2 = np.array([[3575.20789111229, -0.463657413728510, 2062.40502075186],
                               [0, 3575.06428531935, 1542.00719185697],
                               [0.00000000e+00, 0.00000000e+00, 1]])

    dist_coeffs1 = np.array(
        [-0.099178476901251, 0.138102681914255, 6.274110222273973e-04, -2.724380259422465e-04, -0.106909615003580])
    dist_coeffs2 = np.array(
        [-0.099518437619588, 0.157127467456666, 0.001132447921996, 1.331016115699895e-04, -0.268492755010379])
    R = np.array([[0.985210517889707, -0.006972366641296, 0.171206371210053],
                  [0.006583654485686, 0.999974299994116, 0.002838105863397],
                  [-0.171221759519952, -0.001668968153708, 0.985231101626514]])
    T = np.array([-0.2090552072686908, -0.198426078216100e-03, 0.018326765876410498])
    width = 4096
    height = 3000
    '''
    # 图像矫正，使双目相机投影面共面，基线与投影面平行
    # 计算校正变换
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
    print(camera_matrix1)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
                                                      (width, height), R, T, alpha=1)
    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, (width, height), cv2.CV_32FC1)
    return P1, P2,R1,R2, map1x, map1y, map2x, map2y


def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)

def showresult():
    X = [i for i in range(1, len(result) + 1)]
    Y = result

    plt.plot(X, Y)
    plt.show()
