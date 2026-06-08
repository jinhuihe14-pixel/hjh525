import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
import os
from datetime import datetime, timedelta

from config.settings import SENSOR_METRICS, RUL_WARNING_DAYS, RUL_CRITICAL_DAYS, MODEL_DIR


class RULPredictor:
    def __init__(self, method="ensemble"):
        self.method = method
        self.models = {}
        self.health_history = {}
        self.is_fitted = False
        self.poly = PolynomialFeatures(degree=2, include_bias=False)

    def _extract_degradation_features(self, health_series, timestamps=None):
        features = {}
        n = len(health_series)
        if n < 2:
            return features

        health_array = np.array(health_series)
        x = np.arange(n).reshape(-1, 1)

        if timestamps is not None:
            time_diffs = [(t - timestamps[0]).total_seconds() for t in timestamps]
            x = np.array(time_diffs).reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(x, health_array)
        features["degradation_rate"] = float(lr.coef_[0])
        features["intercept"] = float(lr.intercept_)
        features["r_squared"] = float(lr.score(x, health_array))

        features["health_mean"] = float(np.mean(health_array))
        features["health_std"] = float(np.std(health_array))
        features["health_trend"] = float(health_array[-1] - health_array[0])
        features["health_min"] = float(np.min(health_array))
        features["health_max"] = float(np.max(health_array))

        if n >= 10:
            recent = health_array[-10:]
            earlier = health_array[:10] if n >= 20 else health_array[: n // 2]
            features["recent_degradation_rate"] = float(np.mean(recent) - np.mean(earlier))
        else:
            features["recent_degradation_rate"] = features["degradation_rate"] * n / 10

        features["acceleration"] = 0
        if n >= 20:
            mid = n // 2
            first_half = health_array[:mid]
            second_half = health_array[mid:]
            rate1 = (first_half[-1] - first_half[0]) / len(first_half)
            rate2 = (second_half[-1] - second_half[0]) / len(second_half)
            features["acceleration"] = float(rate2 - rate1)

        features["volatility"] = float(np.mean(np.abs(np.diff(health_array))))

        return features

    def add_health_reading(self, equipment_id, health_score, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        if equipment_id not in self.health_history:
            self.health_history[equipment_id] = {"timestamps": [], "health_scores": []}

        self.health_history[equipment_id]["timestamps"].append(timestamp)
        self.health_history[equipment_id]["health_scores"].append(health_score)

        max_history = 10000
        if len(self.health_history[equipment_id]["timestamps"]) > max_history:
            self.health_history[equipment_id]["timestamps"] = self.health_history[equipment_id]["timestamps"][-max_history:]
            self.health_history[equipment_id]["health_scores"] = self.health_history[equipment_id]["health_scores"][-max_history:]

    def predict_rul(self, equipment_id, failure_threshold=0.3, method="linear"):
        if equipment_id not in self.health_history:
            return {"rul_days": None, "confidence": 0, "method": method}

        history = self.health_history[equipment_id]
        health_scores = history["health_scores"]
        timestamps = history["timestamps"]

        n = len(health_scores)
        if n < 5:
            current_health = health_scores[-1]
            if current_health > 0.9:
                estimated_days = 90
            elif current_health > 0.7:
                estimated_days = 30
            elif current_health > 0.5:
                estimated_days = 14
            else:
                estimated_days = 7
            return {
                "rul_days": estimated_days,
                "confidence": 0.3,
                "method": "heuristic",
                "current_health": current_health,
                "failure_threshold": failure_threshold,
            }

        features = self._extract_degradation_features(health_scores, timestamps)
        degradation_rate = features.get("degradation_rate", -0.001)
        current_health = health_scores[-1]

        if method == "linear":
            if degradation_rate >= 0:
                degradation_rate = -0.0001
            days_to_failure = (failure_threshold - current_health) / degradation_rate

            time_span = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 86400
            samples_per_day = max(1, n / (time_span / 86400))
            days_to_failure = max(0, days_to_failure / samples_per_day)

        elif method == "exponential":
            if current_health <= 0 or degradation_rate >= 0:
                days_to_failure = 30
            else:
                decay_rate = abs(degradation_rate) / max(current_health, 0.01)
                days_to_failure = -np.log(failure_threshold / current_health) / max(decay_rate, 0.001)

        elif method == "polynomial":
            x = np.arange(n).reshape(-1, 1)
            y = np.array(health_scores)
            try:
                poly = PolynomialFeatures(degree=2)
                x_poly = poly.fit_transform(x)
                model = Ridge(alpha=1.0)
                model.fit(x_poly, y)

                future_x = np.arange(n, n + 1000).reshape(-1, 1)
                future_poly = poly.transform(future_x)
                future_pred = model.predict(future_poly)

                below_threshold = np.where(future_pred <= failure_threshold)[0]
                if len(below_threshold) > 0:
                    days_to_failure = below_threshold[0] / max(1, n / 30)
                else:
                    days_to_failure = 100
            except Exception:
                days_to_failure = 30
        else:
            days_to_failure = 30

        confidence = min(0.95, 0.5 + n * 0.005)
        confidence *= max(0.5, 1 - abs(features.get("volatility", 0)) * 10)

        if days_to_failure < 0:
            days_to_failure = 0.5

        return {
            "rul_days": round(float(days_to_failure), 2),
            "confidence": round(float(confidence), 4),
            "method": method,
            "current_health": current_health,
            "failure_threshold": failure_threshold,
            "degradation_rate": features.get("degradation_rate", 0),
            "recommended_maintenance_days": max(1, int(days_to_failure * 0.7)),
        }

    def predict_rul_ensemble(self, equipment_id, failure_threshold=0.3):
        methods = ["linear", "exponential", "polynomial"]
        predictions = []
        weights = []

        for method in methods:
            result = self.predict_rul(equipment_id, failure_threshold, method)
            if result["rul_days"] is not None:
                predictions.append(result["rul_days"])
                weights.append(result["confidence"])

        if not predictions:
            return {"rul_days": None, "confidence": 0}

        total_weight = sum(weights) if sum(weights) > 0 else 1
        weighted_rul = sum(p * w for p, w in zip(predictions, weights)) / total_weight
        avg_confidence = sum(weights) / len(weights)

        current_health = self.health_history.get(equipment_id, {}).get("health_scores", [0])[-1]

        if weighted_rul <= RUL_CRITICAL_DAYS:
            status = "critical"
        elif weighted_rul <= RUL_WARNING_DAYS:
            status = "warning"
        else:
            status = "normal"

        return {
            "rul_days": round(float(weighted_rul), 2),
            "confidence": round(float(avg_confidence), 4),
            "status": status,
            "current_health": round(float(current_health), 4),
            "method_predictions": dict(zip(methods, predictions)),
            "recommended_maintenance_window": {
                "earliest": (datetime.now() + timedelta(days=max(1, int(weighted_rul * 0.5)))).strftime("%Y-%m-%d"),
                "latest": (datetime.now() + timedelta(days=int(weighted_rul * 0.9))).strftime("%Y-%m-%d"),
            },
        }

    def predict_all(self):
        results = {}
        for eq_id in self.health_history:
            results[eq_id] = self.predict_rul_ensemble(eq_id)
        return results

    def fit_from_dataframe(self, df):
        equipment_groups = df.groupby("equipment_id")
        for eq_id, group in equipment_groups:
            group_sorted = group.sort_values("timestamp")
            for _, row in group_sorted.iterrows():
                self.add_health_reading(
                    eq_id,
                    row.get("health_score", 0.8),
                    row.get("timestamp"),
                )
        self.is_fitted = True
        return self

    def save(self, path):
        data = {
            "health_history": self.health_history,
            "method": self.method,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.health_history = data["health_history"]
        self.method = data["method"]
        self.is_fitted = data["is_fitted"]
        return self
