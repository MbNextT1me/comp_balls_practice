import cv2
import numpy as np
from skimage.measure import label

cam = cv2.VideoCapture("balls.mp4")
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

while cam.isOpened():
    _, frame = cam.read()
    if _:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, t = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mask = np.ones((6, 6))
        t = cv2.erode(t, mask, iterations=2)
        labeled = label(t)
        
        cv2.putText(frame, f"Balls on image: {labeled.max()}", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        cv2.imshow('Camera', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()