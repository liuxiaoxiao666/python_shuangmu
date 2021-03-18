#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图



import numpy as np
#z:最值差 x:x轴 y:z轴  var:方差
z = np.linspace(0,0.2,1000)
var=np.linspace(0,3,1000)
x = np.linspace(-100,400,1000)
y = np.linspace(500,1000,1000)
vard=[0.645737683,0.516380823,1.076775001,0.258464122,0.202408539,0.869480739,0.484136411,0.108245668,0.102656791,0.200098697,1.221993488,0.80038074,0.804468416,0.261622517,0.439005183,2.838191193]
zd = [0.10287606,0.084813048,0.130491285,0.07842602,0.109017163,0.097299482,0.087445149,0.043711018,0.0391548,0.061290265,0.126167188,0.137795939,0.107734509,0.094141791,0.097676874,0.175210245]
xd = [-68.33462548,94.28581164,283.4427481,-87.84607571,-86.5513526,105.7878688,317.5644423,338.6688965,159.942583,-68.24552864,-67.11650749,173.6914588,368.7724962,375.7570151,168.5721369,-26.64892386]
yd = [486.3038826,529.6262806,527.995503,536.4142519,630.5534974,601.3320135,602.0402306,710.7566396,728.8386738,741.4241424,840.2500951,808.9210363,811.0038828,970.870093,984.9856916,983.6156721]
print(zd)
ax1.scatter3D(xd,yd,vard, cmap='Blues')  #绘制散点图
plt.savefig("fangcha.png")
plt.show()
