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


def main():
    filecounter=0
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_device('841612070330')
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared,1,1280,720,rs.format.y8,30)
    config.enable_stream(rs.stream.infrared,2,1280,720,rs.format.y8,30)

    # Start streaming
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()

    # 对齐
    align_to = rs.stream.color
    align = rs.align(align_to)

    print(intr)  # 获取内参 width: 640, height: 480, ppx: 319.115, ppy: 234.382, fx: 597.267, fy: 597.267, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]
    print(intr.ppx)  # 获取指定某个内参

    window = 'Camera'
    cv2.namedWindow(window)
    at_detector = apriltag.Detector(families='tag36h11')  # for windows

    last_distance = 0
    distance_map = []
    error_value_map = []
    error_pct_map = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)  # 对齐
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            '''获取红外影像'''
            ir_frame_left=frames.get_infrared_frame(1)
            ir_frame_right = frames.get_infrared_frame(2)
            dpt_frame = depth_frame.as_depth_frame()

            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            """获取红外摄像头的图像数据"""
            ir_left_image = np.asanyarray(ir_frame_left.get_data())
            ir_right_image = np.asanyarray(ir_frame_right.get_data())

            images2 = np.hstack((ir_left_image, ir_right_image))
            cv2.imshow("Display pic_irt", images2)

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
                    last_distance = distance

                    '''直接获取realsense距离'''
                    depth_dis = dpt_frame.get_distance(*list(point[0]))


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
