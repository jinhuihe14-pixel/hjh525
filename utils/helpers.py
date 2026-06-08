import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_timestamp_series(start_time, n_points, interval_sec=1):
    timestamps = []
    for i in range(n_points):
        ts = start_time + timedelta(seconds=i * interval_sec)
        timestamps.append(ts)
    return timestamps


def add_noise(data, noise_level=0.02):
    if isinstance(data, np.ndarray):
        noise = np.random.normal(0, noise_level * np.abs(data), data.shape)
        return data + noise
    elif isinstance(data, pd.Series):
        noise = np.random.normal(0, noise_level * data.abs(), len(data))
        return data + noise
    else:
        return data * (1 + np.random.normal(0, noise_level))


def calculate_statistics(data):
    if len(data) == 0:
        return {}
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "range": float(np.ptp(arr)),
        "count": len(arr),
    }


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def format_currency(amount):
    if amount >= 10000:
        return f"¥{amount/10000:.2f}万"
    else:
        return f"¥{amount:,.2f}"
