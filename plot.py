import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_column(file_path: str, column_name: str):
    if not file_path.lower().endswith('.csv'):
        return
    
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        return
    
    y_raw = pd.to_numeric(df[column_name], errors="coerce")
    x_raw = np.arange(len(y_raw))
    mask = np.isfinite(y_raw) # Remove NaN/inf for fitting
    x = x_raw[mask]
    y = y_raw[mask]
    
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x_raw + intercept # Trend for full x-range
    plt.figure()
    plt.plot(x_raw, y_raw, label=column_name)

    plt.plot(x_raw, trend, label="Trend", linewidth=2, color="black", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel(column_name)
    plt.title(f"{column_name} over episodes")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.01)

plot_column(os.path.join(os.getcwd(), 'Results_DrQv2_Mujoco_Pick_Incomplete', 'eval.csv'), 'episode_reward')
plot_column(os.path.join(os.getcwd(), 'Results_DrQv2_Metaworld_Reach', 'eval.csv'), 'episode_reward')
plt.show()