import socket,cv2, pickle,struct
import PID
import numpy as np
from piservo import Servo

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = "192.168.27.252" 
port = 9999
client_socket.connect((host_ip,port)) 
data = b""
payload_size = struct.calcsize("Q")
cx = None
cy = None

x_axis = Servo(12)
y_axis = Servo(13)

x_axis.write(90)
y_axis.write(90)

pan = 90
tilt = 90
width = 320
height = 240

P1 = 1
I1 = 0
D1 = 0.2
pan_target = width/2
pid1 = PID.PID(P1, I1, D1)
pid1.SetPoint = pan_target
pid1.setSampleTime(1)
P2 = 1
I2 = 0
D2 = 0.2
tilt_target = height/2
pid2 = PID.PID(P2, I2, D2)
pid2.SetPoint = tilt_target
pid2.setSampleTime(1)

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min
def track(frame):
    global cx,cy
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #(hMin = 20 , sMin = 127, vMin = 115), (hMax = 41 , sMax = 255, vMax = 199)
    #(hMin = 9 , sMin = 80, vMin = 128), (hMax = 52 , sMax = 255, vMax = 255)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    # lower_yellow = np.array([9, 80, 128])
    # upper_yellow = np.array([52, 255, 255])


    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    img_erosion = cv2.erode(blur, np.ones((5, 5)), iterations=2)
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
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024) 
        if not packet: break
        data+=packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q",packed_msg_size)[0]
    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    data  = data[msg_size:]
    frame = pickle.loads(frame_data)
    (h, w) = frame.shape[:2]
    tracked_frame = track(frame)
    center_point  = int(h/2) , int(w/2)
    
    if cx is not None:
        pid1.update(cx)
        target_pwm1 = pid1.output
        target_pwm1 = map_range(target_pwm1,-w,w,40,140)
        pid2.update(cy)
        target_pwm2 = pid2.output
        target_pwm2 = map_range(target_pwm2, h, -h, 40, 140)
        x_axis.write(int(target_pwm1))
        y_axis.write(int(target_pwm2))
        print(int(target_pwm1),int(target_pwm2))
        

    if cx is not None and cx > center_point[1] - 20 and cx < center_point[1] + 20 and cy > center_point[0] - 20 and cy < center_point[0] + 20:
            cv2.imwrite("picture.jpg",frame)
            
    cv2.imshow("RECEIVING VIDEO",frame)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
client_socket.close()

