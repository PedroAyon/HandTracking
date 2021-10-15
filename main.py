import cv2
import time
import HandTrackingModule as htm

p_time = 0
c_time = 0
w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, w_cam)
cap.set(4, h_cam)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)
    if len(landmark_list) != 0:
        print(landmark_list[4])

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

    cv2.imshow('Image', img)
    cv2.waitKey(1)
