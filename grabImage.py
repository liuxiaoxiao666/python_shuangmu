import numpy as np
import PySpin
import threading
import sys
import cv2
import ctypes
import inspect
from TagsInfo.gui import video
from TagsInfo.Global_variable import *
from TagsInfo.calculate import calculateThread

import time
image_width=4096
image_height=3000
class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2


image_camera0=0
image_camera1=0
CHOSEN_TRIGGER = TriggerType.HARDWARE
cam_list=[]
camp1=1
camp2=2
cam=1
system=0
dllPath = "./sspusdk/sspuWrap.dll"
dll = ctypes.cdll.LoadLibrary(dllPath)
cameraSerialNumber = ["17023542", "17491224"]

def sspu_start():
    dll.sspu_start()

def sspu_stop():
    dll.sspu_stop()

def camParam_Initialize(camID):
    """
    :param camID: 主相机的编号，如果是0，那么左相机为主相机
    :return: return init successfully or not
    """

    result=True
    '''#############[SSPU INIT START]###################'''
    sspu_start()
    global system,cam_list
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()


    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)
    # Finish if there are no cameras
    if num_cameras ==0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False
    # 根据相机序列号判断是否需要进行更换，左相机为主相机
    global camp1,camp2

    nodemap_tldevice = cam_list[0].GetTLDeviceNodeMap()
    node_device_vendor_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
    device_vendor_name="17023542"
    if PySpin.IsAvailable(node_device_vendor_name) and PySpin.IsReadable(node_device_vendor_name):
        device_vendor_name = node_device_vendor_name.ToString()
    print(device_vendor_name)
    camp1 = cam_list[0]
    camp2 = cam_list[1]
    if device_vendor_name !=cameraSerialNumber[0]:
        camp1 = cam_list[1]
        camp2 = cam_list[0]
    # Run example on each camera

    for i, cam in enumerate(cam_list):
        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            ############ Initialize camera################
            cam.Init()
            ############ Retrieve GenICam nodemap#########
            nodemap = cam.GetNodeMap()
            ##############Configure trigger###############
            # Ensure trigger mode off
            # The trigger must be disabled in order to configure whether the source
            # is software or hardware.
            nodemap = cam.GetNodeMap()
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
                print('Unable to disable trigger mode (node retrieval). Aborting...')
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
            if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
                print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

            print('Trigger mode disabled...')
            # Set TriggerSelector to FrameStart
            # For this example, the trigger selector should be set to frame start.
            # This is the default for most cameras.
            node_trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
            if not PySpin.IsAvailable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
                print('Unable to get trigger selector (node retrieval). Aborting...')
                return False

            node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
            if not PySpin.IsAvailable(node_trigger_selector_framestart) or not PySpin.IsReadable(
                    node_trigger_selector_framestart):
                print('Unable to set trigger selector (enum entry retrieval). Aborting...')
                return False
            node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())

            print('Trigger selector set to frame start...')

            # Select trigger source
            # The trigger source must be set to hardware or software while trigger
            # mode is off.
            node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
            if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
                print('Unable to get trigger source (node retrieval). Aborting...')
                return False
            if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
                node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
                if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                        node_trigger_source_software):
                    print('Unable to set trigger source (enum entry retrieval). Aborting...')
                    return False
                node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
                print('Trigger source set to software...')

            elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
                node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
                if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                        node_trigger_source_hardware):
                    print('Unable to set trigger source (enum entry retrieval). Aborting...')
                    return False
                node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
                print('Trigger source set to hardware...')

            # Turn trigger mode on
            # Once the appropriate trigger source has been set, turn trigger mode
            # on in order to retrieve images using the trigger.
            node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
            if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
                print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
            print('Trigger mode turned back on...')
            #Turn off gain off
            cam.GainAuto.SetValue(PySpin.GainAuto_Once)
            # set gain to 10.5db
            # cam.Gain.SetValue(35)

            # cam.BlackLevel.SetValue(5.86)

            # Set the absolute value of gamma to 1.5
            # cam.Gamma.SetValue(1.25)
            # Turn off auto exposure 自动曝光
            # cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Once)

            # Set exposure mode to "Timed"
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            exposure_time_to_set = 13500
            exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
            cam.ExposureTime.SetValue(exposure_time_to_set)
            print('Shutter time set to %s us...\n' % exposure_time_to_set)
            # cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
            # cam.ExposureTime.SetValue(4000)

            cam.Width.SetValue(image_width)
            cam.Height.SetValue(image_height)
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            cam.BeginAcquisition()
        except PySpin.SpinnakerException as ex:
            print('Error:%s'%ex)
            result=False
    return result


def system_release():
    for cam in cam_list:
        # Deinitialize camera
        cam.DeInit()
    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()
    try:
        dll.sspu_stop()
        print("Trigger stop...")
    except Exception:
        print("Trigger stop exception")
        return False
    # Release system instance
    system.ReleaseInstance()

bdflag=0
cnt=26
def getNextImage_cam0():
    # print("Gain", cam.Gain.GetValue())
    # print("Exposure time", cam.ExposureTime.GetValue())
    # global camp1
    global bdflag,cnt,img_list,camp1

    while True:
        try:
            image_result=camp1.GetNextImage(1000)
            #  Ensure image completion
            image_data=np.ndarray([0])
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                # Getting the image data as a numpy array
                image_data = image_result.GetNDArray()
            img_list['img1']=image_data.copy()
            # print(img1)
            # print("getpic0")

            '''
            cv2.namedWindow('camera0', cv2.WINDOW_NORMAL)
            cv2.imshow("camera0", img1)
            #按esc保存图片
            if cv2.waitKey(2) == 27 and bdflag==0:
                cv2.imwrite(str(cnt)+"cam1.bmp",img1)

                bdflag =1
            elif bdflag==2:
                cv2.imwrite(str(cnt)+"cam1.bmp",img1)
                cnt+=1
                bdflag = 0
            '''
        # return image_data
        except:
            pass

def getNextImage_cam1():
    global bdflag,cnt,img_list,camp2
    while True:
        try:
            image_result=camp2.GetNextImage(1000)
            #  Ensure image completion
            image_data = np.ndarray([0])
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                # Getting the image data as a numpy array
                image_data = image_result.GetNDArray()
            img_list['img2']=image_data.copy()
            # print("getpic1")
            '''
            cv2.namedWindow('camera1', cv2.WINDOW_NORMAL)
            cv2.imshow("camera1", img2)

            if cv2.waitKey(2) == 27 and bdflag==0:
                cv2.imwrite(str(cnt)+"cam2.bmp",img2)

                bdflag=2
            elif bdflag==1:
                cv2.imwrite(str(cnt)+"cam2.bmp",img2)
                cnt+=1
                bdflag=0
            '''
        # return image_data
        except:
            pass

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
def main():
    ##################[camera parameter initialization]#######################
    if camParam_Initialize(0):
        print("Camera initialize success...")
    else:
        print("Camera initialize error...")
    global img_list
    pic_cam0_thread=threading.Thread(target=getNextImage_cam0)
    pic_cam0_thread.start()
    pic_cam1_thread=threading.Thread(target=getNextImage_cam1)
    pic_cam1_thread.start()

    calculate = threading.Thread(target=calculateThread)
    calculate.start()
    # t=time.time()
    # while(1):
    #     print("time cost:",time.time()-t)
    #     t=time.time()
        # image=getNextImage_cam0()
        # image1=getNextImage_cam1()
        # cv2.imshow("camera0",image)
        # cv2.imshow("camera1", image1)
        # cv2.waitKey(200)
        # cv2.imshow("camera1",getNextImage_cam1())

    video()
    Stop_thread(pic_cam1_thread)
    Stop_thread(pic_cam0_thread)
    Stop_thread(calculate)
    # system_release()


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
