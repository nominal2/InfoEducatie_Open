from piservo import Servo
import time

x_axis = Servo(12)
y_axis = Servo(13)

x_axis.write(90)
y_axis.write(90)
time.sleep(2)

while True:

    # Move the servo from 40 to 140 degrees
    for angle in range(40, 141,5):
        x_axis.write(angle)
        time.sleep(0.1)  # Add a small delay for smoother movement
        

    # Move the servo back from 140 to 40 degrees
    for angle in range(140, 39, -5):
        x_axis.write(angle)
        time.sleep(0.1)
        
    
        

x_axis.stop()
y_axis.stop()
