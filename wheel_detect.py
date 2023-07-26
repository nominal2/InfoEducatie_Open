import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cx = None
cy = None
def track(frame):
    global cx,cy
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #(hMin = 20 , sMin = 127, vMin = 115), (hMax = 41 , sMax = 255, vMax = 199)
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([40, 255, 255])
    lower_yellow = np.array([20, 127, 115])
    upper_yellow = np.array([40, 255, 199])


    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

    return frame


while True:
    ret, frame = cap.read()
    frame= cv2.flip(frame, 1)
    if not ret:
        break

    tracked_frame = track(frame)
    #print(cx,cy)
    (h, w) = tracked_frame.shape[:2]
    center_point  = int(h/2) , int(w/2)
    print(center_point)
    cv2.imshow('Tracked', tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
