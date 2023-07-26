import cv2
import numpy as np
from simple_pid import PID

cap = cv2.VideoCapture(0)
cx = None
cy = None

pan = 90
tilt = 90


def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min
def track(frame):
    global cx,cy
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #(hMin = 20 , sMin = 127, vMin = 115), (hMax = 41 , sMax = 255, vMax = 199)
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([40, 255, 255])
    lower_yellow = np.array([20, 127, 115])
    upper_yellow = np.array([40, 255, 199])


    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    img_erosion = cv2.erode(blur, np.ones((5, 5)), iterations=1)
    mask = cv2.dilate(img_erosion, np.ones((3, 3)), iterations=5)
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
    #print(center_point)
    if cx is not None:
        if (cx > center_point[1] + 20):
            pan += 10
            if pan > 140:
                pan = 140

        if (cx < center_point[1] - 20):
            pan -= 10
            if pan < 40:
                pan = 40

        if (cy > center_point[0] + 20):
            tilt += 10
            if tilt > 140:
                tilt = 140

        if (cy < center_point[0] - 20):
            tilt -= 10
            if tilt < 40:
                tilt = 40
    if cx is not None and cx > center_point[1] - 20 and cx < center_point[1] + 20 and cy > center_point[0] - 20 and cy < center_point[0] + 20:
            cv2.imwrite("picture.jpg",frame)
    print(pan, tilt)
    cv2.imshow('Tracked', tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
