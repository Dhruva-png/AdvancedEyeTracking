import cv2
import time
import numpy as np
from utils.data_logger import DataLogger
from utils.gaze_tools import estimate_gaze_direction


class EyeTracker:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.logger = DataLogger()


    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)


            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.2, 5)


                for (ex, ey, ew, eh) in eyes:
                    cx = x + ex + ew//2
                    cy = y + ey + eh//2


                gaze = estimate_gaze_direction(roi_gray, (ex, ey, ew, eh))


                self.logger.log(cx, cy, gaze)
                cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)


            cv2.imshow("Eye Tracking", frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cam.release()
        cv2.destroyAllWindows()