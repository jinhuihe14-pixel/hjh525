import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config.settings import SENSOR_METRICS, WINDOW_SIZE, WINDOW_STEP


class DataPreprocessor:
    def __init__(self, method="standard"):
        self.method = method
        self.scaler = None
        self._metric_stats = {}

    def denoise(self, data, metric, method="kalman"):
        values = data[metric].values
        if method == "moving_average":
            window = 5
            kernel = np.ones(window) / window
            denoised = np.convolve(values, kernel, mode="same")
        elif method == "median":
            denoised = signal.medfilt(values, kernel_size=5)
        elif method == "kalman":
            denoised = self._kalman_filter(values)
        elif method == "wavelet":
            denoised = self._wavelet_denoise(values)
        else:
            denoised = values
        return denoised

    def _kalman_filter(self, data, process_noise=1e-4, measurement_noise=0.1):
        n = len(data)
        x_hat = np.zeros(n)
        P = np.zeros(n)
        x_hat_minus = np.zeros(n)
        P_minus = np.zeros(n)
        K = np.zeros(n)

        x_hat[0] = data[0]
        P[0] = 1.0

        for k in range(1, n):
            x_hat_minus[k] = x_hat[k - 1]
            P_minus[k] = P[k - 1] + process_noise
            K[k] = P_minus[k] / (P_minus[k] + measurement_noise)
            x_hat[k] = x_hat_minus[k] + K[k] * (data[k] - x_hat_minus[k])
            P[k] = (1 - K[k]) * P_minus[k]

        return x_hat

    def _wavelet_denoise(self, data, level=3):
        from scipy import signal as sig
        b, a = sig.butter(4, 0.3, btype="low")
        return sig.filtfilt(b, a, data)

    def denoise_all(self, df, method="kalman"):
        result = df.copy()
        for metric in SENSOR_METRICS:
            if metric in result.columns:
                result[metric] = self.denoise(result, metric, method)
        return result

    def fit_scaler(self, df):
        numeric_cols = SENSOR_METRICS
        data = df[numeric_cols].values
        if self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(data)
        for i, col in enumerate(numeric_cols):
            self._metric_stats[col] = {
                "mean": float(np.mean(data[:, i])),
                "std": float(np.std(data[:, i])),
                "min": float(np.min(data[:, i])),
                "max": float(np.max(data[:, i])),
            }
        return self

    def normalize(self, df):
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        result = df.copy()
        numeric_cols = [c for c in SENSOR_METRICS if c in result.columns]
        data = result[numeric_cols].values
        scaled = self.scaler.transform(data)
        for i, col in enumerate(numeric_cols):
            result[col] = scaled[:, i]
        return result

    def inverse_normalize(self, df):
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        result = df.copy()
        numeric_cols = [c for c in SENSOR_METRICS if c in result.columns]
        data = result[numeric_cols].values
        unscaled = self.scaler.inverse_transform(data)
        for i, col in enumerate(numeric_cols):
            result[col] = unscaled[:, i]
        return result

    def sliding_windows(self, df, window_size=None, step=None):
        window_size = window_size or WINDOW_SIZE
        step = step or WINDOW_STEP
        numeric_cols = [c for c in SENSOR_METRICS if c in df.columns]
        data = df[numeric_cols].values
        n = len(data)
        windows = []
        window_info = []

        for i in range(0, n - window_size + 1, step):
            window = data[i : i + window_size]
            windows.append(window)
            window_info.append(
                {
                    "start_idx": i,
                    "end_idx": i + window_size - 1,
                    "start_time": df["timestamp"].iloc[i] if "timestamp" in df.columns else None,
                    "end_time": df["timestamp"].iloc[i + window_size - 1] if "timestamp" in df.columns else None,
                }
            )

        return np.array(windows), window_info

    def extract_features(self, windows):
        features = []
        for w in windows:
            feat = []
            for i in range(w.shape[1]):
                col = w[:, i]
                feat.extend(
                    [
                        np.mean(col),
                        np.std(col),
                        np.max(col),
                        np.min(col),
                        np.median(col),
                        np.percentile(col, 25),
                        np.percentile(col, 75),
                        np.ptp(col),
                        np.sqrt(np.mean(col ** 2)),
                        np.mean(np.abs(np.diff(col))),
                    ]
                )
            features.append(feat)
        return np.array(features)

    def get_metric_stats(self):
        return self._metric_stats


def preprocess_pipeline(raw_df, preprocessor=None, fit=True):
    if preprocessor is None:
        preprocessor = DataPreprocessor(method="standard")

    if fit:
        preprocessor.fit_scaler(raw_df)

    denoised = preprocessor.denoise_all(raw_df, method="kalman")
    normalized = preprocessor.normalize(denoised)

    return normalized, preprocessor
