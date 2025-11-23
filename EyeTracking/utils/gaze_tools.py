import cv2
import time
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Paths
DATA_PATH = "data/gaze_data.csv"


class EyeTracker:
    def __init__(self, log_data=True, smoothing_factor=0.30):
        """
        High-level eye tracking helper.
        Handles:
        - Face + eye detection
        - Pupil localization
        - Gaze point calculation
        - Data smoothing
        - Data logging
        """

        # Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        # Logging options
        self.log_data = log_data
        self.smoothing_factor = smoothing_factor
        self.prev_point = None

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        # Create CSV if not present
        if not os.path.exists(DATA_PATH):
            df = pd.DataFrame(columns=["timestamp", "x", "y", "eye"])
            df.to_csv(DATA_PATH, index=False)


    def _smooth_point(self, point):
        """Applies exponential smoothing to reduce jitter."""
        if self.prev_point is None:
            self.prev_point = point
            return point

        smoothed = (
            self.smoothing_factor * np.array(point)
            + (1 - self.smoothing_factor) * np.array(self.prev_point)
        )

        self.prev_point = tuple(smoothed.astype(int))
        return self.prev_point


    def _log_to_csv(self, x, y, eye_side):
        """Append gaze data to CSV."""
        if not self.log_data:
            return

        timestamp = time.time()

        df = pd.DataFrame([{
            "timestamp": timestamp,
            "x": int(x),
            "y": int(y),
            "eye": eye_side
        }])

        df.to_csv(DATA_PATH, mode="a", index=False, header=False)


    def detect_gaze(self, frame):
        """
        Detects eyes and returns two things:
        - frame_with_visuals (processed image)
        - gaze_points = {"left": (x,y), "right": (x,y)}
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        gaze_points = {"left": None, "right": None}

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                # Decide whether this is left or right eye based on position
                eye_side = "left" if ex < w // 2 else "right"

                # Crop the eye region
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Threshold to find pupil
                _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)

                # Largest contour = pupil
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    (px, py), radius = cv2.minEnclosingCircle(c)

                    global_x = x + ex + int(px)
                    global_y = y + ey + int(py)

                    # Smooth gaze
                    smoothed = self._smooth_point((global_x, global_y))

                    gaze_points[eye_side] = smoothed

                    # Draw visuals
                    cv2.circle(frame, smoothed, 5, (0, 0, 255), -1)
                    cv2.putText(frame, eye_side.upper(), (global_x - 10, global_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Log to CSV
                    self._log_to_csv(smoothed[0], smoothed[1], eye_side)

        return frame, gaze_points


def get_latest_gaze():
    """
    Returns the latest gaze point from CSV.
    Useful for the dashboard.
    """

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return None

    last = df.tail(1).iloc[0]
    return {
        "x": int(last["x"]),
        "y": int(last["y"]),
        "eye": last["eye"],
        "time": last["timestamp"]
    }
