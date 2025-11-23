import tkinter as tk
from heatmap_generator import generate_heatmap
from tracker import EyeTracker
from utils.data_logger import DataLogger
import threading


def launch_dashboard():
    root = tk.Tk()
    root.title("Eye Tracking Dashboard")
    root.geometry("600x400")


    tracker = EyeTracker()
    logger = DataLogger()


    def start_tracking():
        threading.Thread(target=tracker.run).start()


    def export_heatmap():
        generate_heatmap(logger.file_path)


    tk.Button(root, text="Start Eye Tracking", command=start_tracking, width=30).pack(pady=20)
    tk.Button(root, text="Generate Heatmap", command=export_heatmap, width=30).pack(pady=20)
    tk.Button(root, text="Exit", command=root.quit, width=30).pack(pady=20)


    root.mainloop()