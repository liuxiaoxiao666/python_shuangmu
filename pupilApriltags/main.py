import cv2
import numpy as np
import pupil_apriltags as apriltag
from pupilApriltags.tools import *

F_X = 600
F_Y = 600
C_X = 640 * 0.5  # 默认值(image.w * 0.5)
C_Y = 480 * 0.5  # 默认值(image.h * 0.5)
TAG_SIZE = 0.166


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    fps = 24
    window = 'Camera'
    cv2.namedWindow(window)
    at_detector = apriltag.Detector(families='tag36h11')  # for windows

    while cap.grab():
        success, frame = cap.retrieve()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = at_detector.detect(gray, estimate_tag_pose=True, camera_params=[F_X, F_Y, C_X, C_Y], tag_size=TAG_SIZE)
        show = None
        if (len(detections) == 0):
            show = frame
        else:
            show = frame
            print("%d apriltags have been detected." % len(detections))
            for tag in detections:
                cv2.circle(show, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
                cv2.circle(show, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2)  # right-top
                cv2.circle(show, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2)  # right-bottom
                cv2.circle(show, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom

                translation = tag.pose_t
                # 3*3
                rotation = tag.pose_R
                F = np.array([1, 0, 0,
                              0, -1, 0,
                              0, 0, 1]).reshape((3, 3))

                # fixed_rot = F * rotation
                fixed_rot = np.dot(F, rotation)
                yaw, pitch, roll = wRo_to_euler(fixed_rot)
                print('distance={}m x={} y={} z={} yaw={} pitch{} roll={}'
                      .format(np.linalg.norm(translation), translation[0], translation[1], translation[2],
                              yaw, pitch, roll))

        ########################
        # num_detections = len(detections)
        cv2.imshow(window, show)
        k = cv2.waitKey(1000//int(fps))

        if k == 27:
            break
    cap.release()


if __name__ == '__main__':
    main()