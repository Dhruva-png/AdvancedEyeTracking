"""
advanced_eye_tracking_with_heatmap.py

Features:
- MediaPipe Face Mesh for accurate eye/iris detection
- Blink detection (EAR)
- Gaze direction estimation
- Frame-by-frame logging
- Live heatmap/dashboard (matplotlib) toggleable with 'h'
- Export logs to CSV + Excel (raw logs + heatmap matrix) on exit or with 'e'
- Saves heatmap PNG on export/exit

Keys:
- H : toggle dashboard (heatmap + blink/time series)
- E : export/save right now (CSV, Excel, heatmap PNG)
- Q : quit program and auto-save

Requires:
pip install opencv-python mediapipe numpy pandas matplotlib openpyxl
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

# -------------------------------
# User-configurable parameters
# -------------------------------
LOG_INTERVAL = 0.05         # seconds between logs (approx 20Hz)
HEATMAP_GRID = (64, 36)     # heatmap grid (width_cells, height_cells)
HEATMAP_SPREAD_SIGMA = 1.2  # gaussian spread to apply to each fixation
BLINK_THRESHOLD = 0.20      # EAR threshold for blink detection
SMOOTH_ALPHA = 0.25         # smoothing factor for iris centers
HEATMAP_DECAY = 0.0         # if >0, decay heatmap each second (not used by default)
OUTPUT_PREFIX = "eye_tracking_output"  # prefix for saved files

# -------------------------------
# Setup MediaPipe Face Mesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# landmark sets used in prior script
LEFT_EYE = [33, 133, 160, 159, 158, 144]     # approximate eye contour
RIGHT_EYE = [362, 263, 387, 386, 385, 373]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# Helpers
# -------------------------------
def EAR(eye_pts):
    # eye_pts: Nx2 array of contour points in pixel coords [6 points]
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def get_iris_center(landmarks, iris_indices, frame_w, frame_h):
    pts = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in iris_indices])
    center = np.mean(pts, axis=0)
    return tuple(center.astype(int))

def get_gaze_direction(eye_center, eye_box):
    # eye_center: (x,y) in pixels, eye_box: (x,y,w,h)
    x, y = eye_center
    ex, ey, ew, eh = eye_box
    if ew == 0 or eh == 0:
        return "UNKNOWN"
    nx = (x - ex) / float(ew)
    ny = (y - ey) / float(eh)
    # heuristics, tuneable
    if nx < 0.3:
        return "LEFT"
    if nx > 0.7:
        return "RIGHT"
    if ny < 0.35:
        return "UP"
    if ny > 0.75:
        return "DOWN"
    return "CENTER"

def add_point_to_heatmap(heatmap, x_norm, y_norm, grid_w, grid_h, sigma=1.2):
    """x_norm, y_norm in [0..1] relative coords. Adds a gaussian blob centered at that location."""
    gx = int(np.clip(x_norm * grid_w, 0, grid_w - 1))
    gy = int(np.clip(y_norm * grid_h, 0, grid_h - 1))
    heatmap[gy, gx] += 1.0
    if sigma > 0:
        # apply gaussian spread around the whole heatmap for speed we can do a small kernel
        # But simplest approach: after accumulation, we will gaussian_filter the whole heatmap
        pass

# -------------------------------
# Data structures for logging & heatmap
# -------------------------------
log_data = []
last_log_time = time.time()
heatmap = np.zeros((HEATMAP_GRID[1], HEATMAP_GRID[0]), dtype=float)  # [rows, cols] = [h, w]

# smoothing
smooth_left = None
smooth_right = None

# live dashboard control
show_dashboard = False
last_dashboard_update = 0.0
DASH_UPDATE_INTERVAL = 0.2  # seconds

# blink counter
blink_counter = 0

# -------------------------------
# Video capture
# -------------------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("ERROR: Unable to open the camera. Close other apps that use it or select correct device.")

print("Started. Keys: H toggle dashboard | E export/save now | Q quit and save")

# -------------------------------
# Matplotlib setup for dashboard (non-blocking)
# -------------------------------
plt.ion()
fig = None
ax_cam = None
ax_heat = None
ax_blink = None
heat_img = None
time_series_x = []
time_series_blinks = []

def setup_dashboard():
    global fig, ax_cam, ax_heat, ax_blink, heat_img
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_cam, ax_heat, ax_blink = axes
    ax_cam.set_title("Camera")
    ax_heat.set_title("Gaze Heatmap")
    ax_blink.set_title("Blink Count Over Time")
    heat_img = ax_heat.imshow(np.zeros_like(heatmap), cmap=cm.inferno, origin='lower', interpolation='nearest')
    ax_heat.axis('off')
    ax_blink.set_xlim(0, 30)
    ax_blink.set_ylim(0, 10)
    fig.canvas.draw()
    plt.pause(0.001)

def update_dashboard(frame, heatmap_vis, blink_counter):
    global fig, ax_cam, ax_heat, ax_blink, heat_img, time_series_x, time_series_blinks
    if fig is None:
        setup_dashboard()

    ax_cam.clear()
    ax_cam.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax_cam.set_title("Camera")
    ax_cam.axis('off')

    ax_heat.clear()
    heat_img = ax_heat.imshow(heatmap_vis, cmap=cm.inferno, origin='lower', interpolation='nearest')
    ax_heat.set_title("Gaze Heatmap")
    ax_heat.axis('off')

    # update blink time-series
    now = time.time()
    time_series_x.append(now)
    time_series_blinks.append(blink_counter)
    # keep window of last 30 seconds
    window_start = now - 30
    xs = [t - window_start for t in time_series_x if t >= window_start]
    ys = [b for (t, b) in zip(time_series_x, time_series_blinks) if t >= window_start]
    ax_blink.clear()
    ax_blink.plot(xs, ys, '-o')
    ax_blink.set_title("Blink Count Over Time (last 30s)")
    ax_blink.set_xlabel("seconds")
    ax_blink.set_ylabel("blinks")
    fig.canvas.draw()
    plt.pause(0.001)

# -------------------------------
# Main loop
# -------------------------------
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        blink = False
        gaze_direction = "UNKNOWN"

        if result.multi_face_landmarks:
            mesh = result.multi_face_landmarks[0].landmark

            # extract eye contour and iris centers
            left_eye_pts = np.array([(int(mesh[i].x * w), int(mesh[i].y * h)) for i in LEFT_EYE])
            right_eye_pts = np.array([(int(mesh[i].x * w), int(mesh[i].y * h)) for i in RIGHT_EYE])

            left_x, left_y, left_w, left_h = cv2.boundingRect(left_eye_pts)
            right_x, right_y, right_w, right_h = cv2.boundingRect(right_eye_pts)

            left_center = get_iris_center(mesh, LEFT_IRIS, w, h)
            right_center = get_iris_center(mesh, RIGHT_IRIS, w, h)

            # smoothing
            if smooth_left is None:
                smooth_left = left_center
                smooth_right = right_center
            else:
                smooth_left = (
                    int(SMOOTH_ALPHA * left_center[0] + (1 - SMOOTH_ALPHA) * smooth_left[0]),
                    int(SMOOTH_ALPHA * left_center[1] + (1 - SMOOTH_ALPHA) * smooth_left[1])
                )
                smooth_right = (
                    int(SMOOTH_ALPHA * right_center[0] + (1 - SMOOTH_ALPHA) * smooth_right[0]),
                    int(SMOOTH_ALPHA * right_center[1] + (1 - SMOOTH_ALPHA) * smooth_right[1])
                )

            # EAR blink detection
            left_ear = EAR(left_eye_pts)
            right_ear = EAR(right_eye_pts)
            if left_ear < BLINK_THRESHOLD and right_ear < BLINK_THRESHOLD:
                blink = True
                blink_counter += 1

            # gaze (use left eye for simplicity here)
            gaze_direction = get_gaze_direction(smooth_left, (left_x, left_y, left_w, left_h))

            # draw results on frame
            cv2.circle(frame, smooth_left, 4, (0, 0, 255), -1)
            cv2.circle(frame, smooth_right, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (255, 255, 255), 1)
            cv2.rectangle(frame, (right_x, right_y), (right_x + right_w, right_y + right_h), (255, 255, 255), 1)
            cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"Blink Count: {blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # compute normalized gaze point (center between two irises) in [0..1]
            avg_x = (smooth_left[0] + smooth_right[0]) / 2.0
            avg_y = (smooth_left[1] + smooth_right[1]) / 2.0
            x_norm = np.clip(avg_x / float(w), 0.0, 1.0)
            y_norm = np.clip(avg_y / float(h), 0.0, 1.0)

            # accumulate heatmap (count-based)
            gx = int(np.clip(x_norm * HEATMAP_GRID[0], 0, HEATMAP_GRID[0]-1))
            gy = int(np.clip(y_norm * HEATMAP_GRID[1], 0, HEATMAP_GRID[1]-1))
            heatmap[gy, gx] += 1.0

        # logging at fixed intervals
        now = time.time()
        if now - last_log_time >= LOG_INTERVAL:
            ts = datetime.now().isoformat(timespec='milliseconds')
            row = {
                "timestamp": ts,
                "left_x": int(smooth_left[0]) if smooth_left is not None else None,
                "left_y": int(smooth_left[1]) if smooth_left is not None else None,
                "right_x": int(smooth_right[0]) if smooth_right is not None else None,
                "right_y": int(smooth_right[1]) if smooth_right is not None else None,
                "gaze": gaze_direction,
                "blink": blink,
                "blink_count": blink_counter
            }
            log_data.append(row)
            last_log_time = now

        # display camera
        cv2.imshow("Advanced Eye Tracking (q=quit, h=heatmap, e=export)", frame)

        # dashboard update (non-blocking)
        if show_dashboard and (time.time() - last_dashboard_update) > DASH_UPDATE_INTERVAL:
            # generate a smoothed heatmap visualization
            heat_vis = gaussian_filter(heatmap, sigma=HEATMAP_SPREAD_SIGMA)
            heat_vis_norm = (heat_vis / (heat_vis.max() + 1e-9))
            update_dashboard(frame, heat_vis_norm, blink_counter)
            last_dashboard_update = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('h'):
            show_dashboard = not show_dashboard
            if show_dashboard:
                print("Dashboard ON")
                setup_dashboard()
            else:
                print("Dashboard OFF")

        if key == ord('e'):
            print("Manual export requested...")
            # perform export below (reuse export function)
            # we will call same export routine as on-exit
            # fall through to export function below after main loop (we just call inline here)
            pass_export = True
            # call export routine (defined inline below)
            # we call the same code as quitting but we don't exit
            heat_vis = gaussian_filter(heatmap, sigma=HEATMAP_SPREAD_SIGMA)
            heat_vis_norm = heat_vis / (heat_vis.max() + 1e-9)
            # save files
            csv_fname = f"{OUTPUT_PREFIX}_raw_log.csv"
            excel_fname = f"{OUTPUT_PREFIX}_log_and_heatmap.xlsx"
            heatmap_png = f"{OUTPUT_PREFIX}_heatmap.png"
            print(f"Saving CSV -> {csv_fname}")
            pd.DataFrame(log_data).to_csv(csv_fname, index=False)
            print(f"Saving Excel -> {excel_fname} (raw logs + heatmap matrix)")
            # write heatmap matrix as sheet
            writer = pd.ExcelWriter(excel_fname, engine="openpyxl")
            pd.DataFrame(log_data).to_excel(writer, sheet_name="raw_logs", index=False)
            pd.DataFrame(heatmap).to_excel(writer, sheet_name="heatmap_matrix", index=False)
            writer.save()
            plt.imsave(heatmap_png, heat_vis_norm, cmap=cm.inferno)
            print("Manual export complete.")

        if key == ord('q'):
            print("Quit requested. Saving data...")
            break

except KeyboardInterrupt:
    print("Interrupted by user; saving and exiting.")

finally:
    # graceful shutdown
    cam.release()
    cv2.destroyAllWindows()

    # smoothing / normalize heatmap for visualization
    heat_vis = gaussian_filter(heatmap, sigma=HEATMAP_SPREAD_SIGMA)
    heat_vis_norm = heat_vis / (heat_vis.max() + 1e-9)

    # save CSV + Excel + PNG
    csv_fname = f"{OUTPUT_PREFIX}_raw_log.csv"
    excel_fname = f"{OUTPUT_PREFIX}_log_and_heatmap.xlsx"
    heatmap_png = f"{OUTPUT_PREFIX}_heatmap.png"

    print(f"Saving CSV -> {csv_fname}")
    pd.DataFrame(log_data).to_csv(csv_fname, index=False)

    print(f"Saving Excel -> {excel_fname} (raw logs + heatmap matrix + summary)")
    writer = pd.ExcelWriter(excel_fname, engine="openpyxl")
    pd.DataFrame(log_data).to_excel(writer, sheet_name="raw_logs", index=False)
    pd.DataFrame(heatmap).to_excel(writer, sheet_name="heatmap_matrix", index=False)

    # Add a small summary sheet
    summary = {
        "recorded_at": [datetime.now().isoformat()],
        "rows_logged": [len(log_data)],
        "heatmap_cells_x": [HEATMAP_GRID[0]],
        "heatmap_cells_y": [HEATMAP_GRID[1]],
        "total_blinks": [blink_counter]
    }
    pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
    writer.save()

    # save a PNG visualization of the heatmap
    plt.imsave(heatmap_png, heat_vis_norm, cmap=cm.inferno)
    print(f"Saved heatmap image -> {heatmap_png}")

    # show final heatmap briefly
    try:
        import matplotlib.pyplot as plt2
        plt2.figure(figsize=(6, 4))
        plt2.imshow(heat_vis_norm, cmap=cm.inferno, origin='lower')
        plt2.title("Final Gaze Heatmap (normalized)")
        plt2.axis('off')
        plt2.show(block=False)
        plt2.pause(2.0)
        plt2.close()
    except Exception:
        pass

    print("Export complete. Files:")
    print(" -", csv_fname)
    print(" -", excel_fname)
    print(" -", heatmap_png)
    print("Done.")
