import os

import cv2

camera_version = 1

cap = cv2.VideoCapture(camera_version)

while True:
    ret, frame = cap.read()
    cv2.imshow('window', frame)
    pressed_key = cv2.waitKey(25)
    if pressed_key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows