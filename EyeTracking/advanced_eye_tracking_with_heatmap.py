import cv2

#Load Haar cascade files for face + eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)

        for (ex, ey, ew, eh) in eyes:
            # Draw eye box
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Draw tracking point (center of eye)
            cx = ex + ew // 2
            cy = ey + eh // 2
            cv2.circle(roi_color, (cx, cy), 4, (0, 0, 255), -1)

    cv2.imshow("Eye Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()