import tkinter as tk
import cv2
from TagsInfo.Global_variable import *
from TagsInfo.calculate import tans
from TagsInfo.getTags import Tag
from PIL import Image, ImageTk
from tkinter import Canvas, messagebox
import time
import csv

'''可视化布局'''
title = 'Position detection'
window_width = 760
window_height = 720
image_width = 460
image_height = int(image_width*0.75)
imagepos_x = 0
imagepos_y = 0
butpos_x = 450
butpos_y = 450
cntcntc = 1

def startroi():
    global roi_p
    roi_p["initstart"]=True


def savedata():
    try:
        b.config(state=tk.DISABLED)
        global save_p
        save_p['catch_num']=int(V1.get())
        save_p['catch_flag']=True
        save_p['catch_fname'] = V2.get() + '.csv'

    except Exception as e:
        print(e)
        b.config(state=tk.ACTIVE)
        save_p['catch_flag']=False
        messagebox.showwarning('警告', '数据量请输入数字')
    pass
def test_a():
    try:
        ta.config(state=tk.DISABLED)
        global test_area
        test_area['ta_num'] = int(V1.get())
        test_area['ta_flag'] = True
        test_area['ta_fname'] = V2.get() + '.csv'
    except Exception as e:
        print(e)
        ta.config(state=tk.ACTIVE)
        test_area['catch_flag']=False
        messagebox.showwarning('警告', '数据量请输入数字')
    pass

def w_data():
    testmovdist['save']=True
    testmovdist['wd_num'] = int(V1.get())
    testmovdist['wd_fname'] = V2.get() + '.csv'
def video():
    def video_loop():
        try:
            while True:

                global img_list, cntcntc, res, result, shared_lock,save_p

                shared_lock.acquire()
                '''
                if len(img_list['roi1'])==0 or len(img_list['roi2'])==0:
                    image1 = img_list['img1'].copy()
                    image2 = img_list['img2'].copy()
                else:
                    image1=img_list['roi1'].copy()
                    image1=np.array(image1)
                    image2 = img_list['roi2'].copy()
                    image2=np.array(image2)
                '''
                # print("image1.shape:",image1.shape)
                # print("image2.shape:",image2.shape)
                # print("image1:",image1)
                # print("image2:",image2)
                image1 = img_list['img1'].copy()
                image2 = img_list['img2'].copy()
                tag = tagpos.copy()
                shared_lock.release()
                if save_p['finish']:
                    save_p['finish']=False
                    b.config(state=tk.ACTIVE)
                if test_area['finish']:
                    test_area['finish']=False
                    ta.config(state=tk.ACTIVE)
                image1 = cv2.resize(image1, (image_width, image_height))
                pilImage1 = Image.fromarray(image1)
                if tag['tags'] != None and type(tag['tags'][1]) == list:
                    # print(tag['tags'][1][0])  # tag['tags']包含三个，第一个是状态位，第二个保存tag，第三个保存id
                    if type(tag['tags'][1][0]) == Tag:
                        '''str object doesn't has points'''
                        value_tag1_z['text'] = tag['tags'][1][0].points[0][2] * 1000
                        # round(tagpos['dis'][1][0].points[0][2]*1000,15)
                        value_tag1_y['text'] = tag['tags'][1][0].points[0][1] * 1000  # round(tagpos['dis'][1][0].points[0][1]*1000,15)
                        value_tag1_x['text'] = tag['tags'][1][0].points[0][0] * 1000  # round(tagpos['dis'][1][0].points[0][0]*1000,15)
                        value_tag1_rz['text'] = tag['tags'][1][0].rotation[
                            2]  # round(tagpos['dis'][1][0].rotation[2], 15)
                        value_tag1_ry['text'] = tag['tags'][1][0].rotation[
                            1]  # round(tagpos['dis'][1][0].rotation[1], 15)
                        value_tag1_rx['text'] = tag['tags'][1][0].rotation[
                            0]  # round(tagpos['dis'][1][0].rotation[0], 15)
                        # value_tag1_dsv['text']=tag['dis'][1][0].disToSurface
                # label_tag1_x['text']=cntcntc
                cntcntc = cntcntc + 1
                if pilImage1 != None:
                    tkImage1 = ImageTk.PhotoImage(image=pilImage1)
                    canvas2.create_image(0, 0, anchor='nw', image=tkImage1)
                image2 = cv2.resize(image2, (image_width, image_height))
                pilImage2 = Image.fromarray(image2)
                if pilImage2 != None:
                    tkImage2 = ImageTk.PhotoImage(image=pilImage2)
                    canvas4.create_image(0, 0, anchor='nw', image=tkImage2)
                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()
        except Exception as e:
            print(e)

    video_loop()
    win.mainloop()

def testangle():
    testangledist['flagangle']=True
    print("testangle")
def testmov():
    testmovdist['flagmov']=True
    print("testmov")
def dbq_test():
    print("dbq_test")
win = tk.Tk()
win.title(title)
win.geometry(str(window_width) + 'x' + str(window_height))
label_tag1_position = tk.Label(win, text='tag position:', width=15, height=1, anchor='w')
label_tag1_position.place(x=20, y=30)
label_tag1_x = tk.Label(win, text='x:', width=15, height=1, anchor='w')
label_tag1_x.place(x=40, y=60)
# width:显示位数
value_tag1_x = tk.Label(win, text='0', width=25, height=1, anchor='w')
value_tag1_x.place(x=100, y=60)
label_tag1_y = tk.Label(win, text='y:', width=15, height=1, anchor='w')
label_tag1_y.place(x=40, y=90)
value_tag1_y = tk.Label(win, text='0', width=25, height=1, anchor='w')
value_tag1_y.place(x=100, y=90)
label_tag1_z = tk.Label(win, text='z:', width=15, height=1, anchor='w')
label_tag1_z.place(x=40, y=120)
value_tag1_z = tk.Label(win, text='0', width=25, height=1, anchor='w')
value_tag1_z.place(x=100, y=120)

label_tag1_rotation = tk.Label(win, text='tag rotation:', width=15, height=1, anchor='w')
label_tag1_rotation.place(x=20, y=150)
label_tag1_rx = tk.Label(win, text='rx:', width=15, height=1, anchor='w')
label_tag1_rx.place(x=40, y=180)
label_tag1_ry = tk.Label(win, text='ry:', width=15, height=1, anchor='w')
label_tag1_ry.place(x=40, y=210)
label_tag1_rz = tk.Label(win, text='rz:', width=15, height=1, anchor='w')
label_tag1_rz.place(x=40, y=240)
value_tag1_rx = tk.Label(win, text='0', width=25, height=1, anchor='w')
value_tag1_rx.place(x=100, y=180)
value_tag1_ry = tk.Label(win, text='0', width=25, height=1, anchor='w')
value_tag1_ry.place(x=100, y=210)
value_tag1_rz = tk.Label(win, text='0', width=25, height=1, anchor='w')
value_tag1_rz.place(x=100, y=240)
label_tag1_ds = tk.Label(win, text='导出信息:', width=15, height=1, anchor='w')
label_tag1_ds.place(x=20, y=300)
value_tag1_dsv = tk.Label(win, text='数据量：', width=25, height=1, anchor='w')
value_tag1_dsv.place(x=40, y=330)
V1= tk.StringVar()
e1 = tk.Entry(win, show=None, width=15,textvariable = V1)
e1.place(x=100, y=330)
e2t = tk.Label(win, text='文件名：', width=5, height=1, anchor='w')
e2t.place(x=40, y=360)
V2 = tk.StringVar()
e2v = tk.Entry(win, show=None, width=15,textvariable = V2)
e2v.place(x=100, y=360)
b = tk.Button(win, text='保存数据', font=('Arial', 12), width=10, height=1, command=savedata)
b.place(x=70, y=390)
start = tk.Button(win, text='初始化', font=('Arial', 12), width=10, height=1, command=startroi)
start.place(x=70, y=430)
cc = tk.Button(win, text='角度测试', font=('Arial', 12), width=10, height=1, command=testangle)
cc.place(x=70, y=470)
cc = tk.Button(win, text='位移测试', font=('Arial', 12), width=10, height=1, command=testmov)
cc.place(x=70, y=510)
ta = tk.Button(win, text='范围测试', font=('Arial', 12), width=10, height=1, command=test_a)
ta.place(x=70, y=550)
wd = tk.Button(win, text='保存位移数据', font=('Arial', 12), width=10, height=1, command=w_data)
wd.place(x=70, y=590)
dbq = tk.Button(win, text='多标签测试', font=('Arial', 12), width=10, height=1, command=dbq_test)
dbq.place(x=70, y=630)
canvas2 = Canvas(win, bg='white', width=image_width, height=image_height)
canvas2.place(x=295, y=0)
canvas4 = Canvas(win, bg='white', width=image_width, height=image_height)
canvas4.place(x=295, y=image_height)
# video()
