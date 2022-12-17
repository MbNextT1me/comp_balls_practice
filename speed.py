import cv2
import numpy as np
import time
import math

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv2.CAP_PROP_EXPOSURE, -3)

measures = []
hsv = []

x_prv = 0 
y_prv = 0

def contours(m):
    return cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def find_speed(x,y,radius):
    global x_prv
    global y_prv
    global del_time

    ball_size = 54
    
    pix_in_mm = radius*2/ball_size
    distance = (math.sqrt((x - x_prv)**2+(y-y_prv)**2)) / pix_in_mm
    speed = round(distance / (round(time.time() * 1000) - del_time) * 1000 / 100,1)
    del_time = round(time.time() * 1000)

    x_prv = x
    y_prv = y

    if speed != 0.0:
        print(speed)

    cv2.putText(frame, f"Speed: {speed} m/s", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 1,(0, 0, 0))
        

def circle(c):
    if len(c) > 0:
        c = max(c, key=cv2.contourArea)
        (x,y), radius = cv2.minEnclosingCircle(c)

        find_speed(x,y,radius)
        
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), 2)
        colour_arr = hsv[int(y),int(x)]
        if lower_blue[0] <= colour_arr[0] <= upper_blue[0] \
         and lower_blue[1] <= colour_arr[1] <= upper_blue[1] \
         and lower_blue[1] <= colour_arr[1] <= upper_blue[1]:
         if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        
def result(m):
    return cv2.bitwise_and(frame, frame, mask = m)

del_time = round(time.time() * 1000)

while cam.isOpened():
    ret, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60,180,40])
    upper_blue = np.array([120,260,113])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    contours_blue, _ = contours(mask_blue)
    res_b = result(mask_blue)

    circle(contours_blue)

    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()