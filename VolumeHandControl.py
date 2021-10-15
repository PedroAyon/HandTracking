import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, w_cam)
cap.set(4, h_cam)

p_time = 0

detector = htm.HandDetector(min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)
    if len(landmark_list) != 0:
        index_x, index_y = landmark_list[8][1], landmark_list[8][2]
        thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]
        middle_x, middle_y = int((index_x + thumb_x) / 2), int((index_y + thumb_y) / 2)

        cv2.circle(img, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (middle_x, middle_y), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (index_x, index_y), (thumb_x, thumb_y), (255, 0, 0), 3)

        lenght = int(math.hypot(index_x - thumb_x, index_y - thumb_y))

        volume_level = np.interp(lenght, [0, 200], [min_volume, max_volume])
        volume_level_bar = np.interp(lenght, [0, 200], [400, 150])

        volume.SetMasterVolumeLevel(volume_level, None)
        print(lenght, volume_level)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volume_level_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(
        img,
        str(int(fps)),
        (10, 70),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 255),
        3
    )

    cv2.imshow("Img", img)
    cv2.waitKey(1)
