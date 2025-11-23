import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image


def generate_heatmap(csv_path="data/gaze_data.csv",
                     output_path="output/heatmap.png",
                     canvas_size=(1280, 720),
                     cmap="jet"):
    """
    Generates a heatmap from gaze data stored in gaze_data.csv.
    Returns the output_path of the heatmap image.
    """

    if not os.path.exists(csv_path):
        print("[ERROR] No gaze data found. Did you run the eye tracker?")
        return None

    df = pd.read_csv(csv_path)

    if df.empty:
        print("[ERROR] Gaze data file is empty. No heatmap generated.")
        return None

    # Extract gaze coordinates
    x = df["x"].values
    y = df["y"].values

    # Flip y axis to match image coordinates
    y = canvas_size[1] - y

    # Compute KDE (heat density)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xi, yi = np.mgrid[0:canvas_size[0]:canvas_size[0]*1j,
                      0:canvas_size[1]:canvas_size[1]*1j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

    # Create heatmap plot
    plt.figure(figsize=(12.8, 7.2))
    plt.imshow(zi.reshape(canvas_size[1], canvas_size[0]).T,
               origin="lower",
               cmap=cmap,
               alpha=0.75)
    plt.axis("off")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[SUCCESS] Heatmap saved to {output_path}")
    return output_path


def export_to_excel(csv_path="data/gaze_data.csv",
                    output_path="output/gaze_data.xlsx"):
    """
    Exports the gaze data CSV to Excel format.
    """
    if not os.path.exists(csv_path):
        print("[ERROR] No gaze data file found.")
        return None

    df = pd.read_csv(csv_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"[SUCCESS] Excel file saved to {output_path}")
    return output_path


def get_session_statistics(csv_path="data/gaze_data.csv"):
    """
    Returns useful metrics for the dashboard.
    """
    if not os.path.exists(csv_path):
        return {"error": "No gaze data available"}

    df = pd.read_csv(csv_path)

    if df.empty:
        return {"error": "No data recorded"}

    duration = df["timestamp"].max() - df["timestamp"].min()
    num_points = len(df)

    stats = {
        "Total Samples": num_points,
        "Duration (seconds)": round(duration, 2),
        "Average X": round(df["x"].mean(), 2),
        "Average Y": round(df["y"].mean(), 2),
        "Left Eye Samples": len(df[df["eye"] == "left"]),
        "Right Eye Samples": len(df[df["eye"] == "right"])
    }

    return stats
