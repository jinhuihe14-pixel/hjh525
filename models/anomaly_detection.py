import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import pickle
import os

from config.settings import (
    SENSOR_METRICS,
    ANOMALY_THRESHOLD_WARNING,
    ANOMALY_THRESHOLD_CRITICAL,
    MODEL_DIR,
)


class ZScoreAnomalyDetector:
    def __init__(self, threshold_warning=None, threshold_critical=None):
        self.threshold_warning = threshold_warning or ANOMALY_THRESHOLD_WARNING
        self.threshold_critical = threshold_critical or ANOMALY_THRESHOLD_CRITICAL
        self.means_ = None
        self.stds_ = None
        self.is_fitted = False
        self.metrics = SENSOR_METRICS

    def fit(self, df):
        metric_cols = [m for m in self.metrics if m in df.columns]
        self.means_ = df[metric_cols].mean().to_dict()
        self.stds_ = df[metric_cols].std().to_dict()
        self.is_fitted = True
        return self

    def partial_fit(self, df):
        if not self.is_fitted:
            return self.fit(df)

        metric_cols = [m for m in self.metrics if m in df.columns]
        alpha = 0.05
        for col in metric_cols:
            new_mean = df[col].mean()
            new_std = df[col].std()
            self.means_[col] = (1 - alpha) * self.means_.get(col, new_mean) + alpha * new_mean
            self.stds_[col] = (1 - alpha) * self.stds_.get(col, new_std) + alpha * new_std
        return self

    def detect(self, df):
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        metric_cols = [m for m in self.metrics if m in df.columns]
        results = []

        for _, row in df.iterrows():
            z_scores = {}
            max_z = 0
            max_metric = None

            for col in metric_cols:
                mean = self.means_[col]
                std = self.stds_[col]
                if std == 0:
                    z = 0
                else:
                    z = abs((row[col] - mean) / std)
                z_scores[col] = z
                if z > max_z:
                    max_z = z
                    max_metric = col

            if max_z >= self.threshold_critical:
                level = "critical"
                is_anomaly = True
            elif max_z >= self.threshold_warning:
                level = "warning"
                is_anomaly = True
            else:
                level = "normal"
                is_anomaly = False

            results.append(
                {
                    "anomaly_score": round(max_z, 4),
                    "anomaly_level": level,
                    "is_anomaly": is_anomaly,
                    "max_anomaly_metric": max_metric,
                    "z_scores": z_scores,
                }
            )

        return pd.DataFrame(results)

    def detect_single(self, reading_dict):
        df = pd.DataFrame([reading_dict])
        result = self.detect(df)
        return result.iloc[0].to_dict()

    def save(self, path):
        data = {
            "means_": self.means_,
            "stds_": self.stds_,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.means_ = data["means_"]
        self.stds_ = data["stds_"]
        self.threshold_warning = data["threshold_warning"]
        self.threshold_critical = data["threshold_critical"]
        self.is_fitted = data["is_fitted"]
        return self


class IsolationForestAnomalyDetector:
    def __init__(self, contamination=0.05, n_estimators=100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler_means = None
        self.scaler_stds = None
        self.metrics = SENSOR_METRICS

    def _prepare_data(self, df):
        metric_cols = [m for m in self.metrics if m in df.columns]
        data = df[metric_cols].values
        if self.scaler_means is not None:
            means = np.array([self.scaler_means.get(m, 0) for m in metric_cols])
            stds = np.array([self.scaler_stds.get(m, 1) for m in metric_cols])
            data = (data - means) / (stds + 1e-8)
        return data

    def fit(self, df):
        metric_cols = [m for m in self.metrics if m in df.columns]
        self.scaler_means = df[metric_cols].mean().to_dict()
        self.scaler_stds = df[metric_cols].std().to_dict()

        data = self._prepare_data(df)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
        )
        self.model.fit(data)
        return self

    def detect(self, df):
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        data = self._prepare_data(df)
        predictions = self.model.predict(data)
        scores = self.model.decision_function(data)
        normalized_scores = (1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)) * 5

        results = []
        for i, (pred, score) in enumerate(zip(predictions, normalized_scores)):
            is_anomaly = pred == -1
            if score >= ANOMALY_THRESHOLD_CRITICAL:
                level = "critical"
            elif score >= ANOMALY_THRESHOLD_WARNING:
                level = "warning"
            else:
                level = "normal"
                is_anomaly = False

            results.append(
                {
                    "anomaly_score": round(float(score), 4),
                    "anomaly_level": level,
                    "is_anomaly": is_anomaly,
                }
            )

        return pd.DataFrame(results)

    def save(self, path):
        data = {
            "model": self.model,
            "scaler_means": self.scaler_means,
            "scaler_stds": self.scaler_stds,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler_means = data["scaler_means"]
        self.scaler_stds = data["scaler_stds"]
        self.contamination = data["contamination"]
        self.n_estimators = data["n_estimators"]
        return self


class EnsembleAnomalyDetector:
    def __init__(self):
        self.zscore_detector = ZScoreAnomalyDetector()
        self.iforest_detector = IsolationForestAnomalyDetector()
        self.weights = {"zscore": 0.6, "iforest": 0.4}
        self.threshold_warning = ANOMALY_THRESHOLD_WARNING
        self.threshold_critical = ANOMALY_THRESHOLD_CRITICAL

    def fit(self, df):
        self.zscore_detector.fit(df)
        self.iforest_detector.fit(df)
        return self

    def partial_fit(self, df):
        self.zscore_detector.partial_fit(df)
        return self

    def detect(self, df):
        z_result = self.zscore_detector.detect(df)
        if_result = self.iforest_detector.detect(df)

        results = []
        for i in range(len(df)):
            z_score = z_result.iloc[i]["anomaly_score"]
            if_score = if_result.iloc[i]["anomaly_score"]

            combined_score = (
                self.weights["zscore"] * z_score + self.weights["iforest"] * if_score
            )

            if combined_score >= self.threshold_critical:
                level = "critical"
                is_anomaly = True
            elif combined_score >= self.threshold_warning:
                level = "warning"
                is_anomaly = True
            else:
                level = "normal"
                is_anomaly = False

            max_metric = z_result.iloc[i].get("max_anomaly_metric", None)

            results.append(
                {
                    "anomaly_score": round(combined_score, 4),
                    "anomaly_level": level,
                    "is_anomaly": is_anomaly,
                    "max_anomaly_metric": max_metric,
                    "zscore_anomaly": z_result.iloc[i]["is_anomaly"],
                    "iforest_anomaly": if_result.iloc[i]["is_anomaly"],
                }
            )

        return pd.DataFrame(results)

    def detect_single(self, reading_dict):
        df = pd.DataFrame([reading_dict])
        result = self.detect(df)
        return result.iloc[0].to_dict()

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.zscore_detector.save(os.path.join(save_dir, "zscore_detector.pkl"))
        self.iforest_detector.save(os.path.join(save_dir, "iforest_detector.pkl"))

    def load(self, save_dir):
        self.zscore_detector.load(os.path.join(save_dir, "zscore_detector.pkl"))
        self.iforest_detector.load(os.path.join(save_dir, "iforest_detector.pkl"))
        return self
