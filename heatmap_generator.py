import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Generates a heatmap image from logged gaze coordinates
def generate_heatmap(csv_path):
    df = pd.read_csv(csv_path)
    heatmap_data = df[['x','y']]


    plt.figure(figsize=(8,6))
    sns.kdeplot(x=heatmap_data['x'], y=heatmap_data['y'], fill=True, cmap="coolwarm", bw_adjust=0.5)
    plt.title("Eye Tracking Heatmap")
    plt.savefig("output/heatmap.png")
    plt.close()
    print("Heatmap saved to output/heatmap.png")