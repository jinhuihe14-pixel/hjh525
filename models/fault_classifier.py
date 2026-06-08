import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os

from config.settings import SENSOR_METRICS, FAULT_TYPES, FAULT_LOCATIONS, MODEL_DIR


class FaultClassifier:
    def __init__(self, method="random_forest"):
        self.method = method
        self.model = None
        self.label_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.metrics = SENSOR_METRICS
        self.feature_cols = None
        self.is_fitted = False
        self.classes_ = None

    def _extract_features(self, df):
        metric_cols = [m for m in self.metrics if m in df.columns]
        features = pd.DataFrame()

        features["vibration_temp_ratio"] = df.get("vibration_freq", 0) / (df.get("temperature", 1) + 1e-8)
        features["current_voltage_ratio"] = df.get("current", 0) / (df.get("voltage", 1) + 1e-8)
        features["load_speed_ratio"] = df.get("load", 0) / (df.get("rotational_speed", 1) + 1e-8)
        features["temp_load_interaction"] = df.get("temperature", 0) * df.get("load", 0)
        features["noise_vib_ratio"] = df.get("noise_level", 0) / (df.get("vibration_freq", 1) + 1e-8)

        for col in metric_cols:
            features[col] = df[col].values
            features[f"{col}_squared"] = df[col].values ** 2

        return features

    def fit(self, df, fault_labels, location_labels=None):
        X = self._extract_features(df)
        self.feature_cols = list(X.columns)

        y_fault = self.label_encoder.fit_transform(fault_labels)
        self.classes_ = list(self.label_encoder.classes_)

        if self.method == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight="balanced",
            )
        elif self.method == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )

        self.model.fit(X.values, y_fault)
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._extract_features(df)
        if self.feature_cols:
            X = X[self.feature_cols]

        y_pred = self.model.predict(X.values)
        y_proba = self.model.predict_proba(X.values)

        fault_types = self.label_encoder.inverse_transform(y_pred)
        confidences = np.max(y_proba, axis=1)

        results = []
        for i in range(len(df)):
            proba_dict = dict(zip(self.classes_, y_proba[i]))
            results.append(
                {
                    "predicted_fault": fault_types[i],
                    "confidence": round(float(confidences[i]), 4),
                    "fault_probabilities": {k: round(float(v), 4) for k, v in proba_dict.items()},
                }
            )

        return pd.DataFrame(results)

    def predict_single(self, reading_dict):
        df = pd.DataFrame([reading_dict])
        result = self.predict(df)
        return result.iloc[0].to_dict()

    def predict_with_location(self, df):
        fault_results = self.predict(df)
        locations = []

        for i, row in fault_results.iterrows():
            fault = row["predicted_fault"]
            if fault == "bearing_wear":
                possible_locs = ["bearing_A", "bearing_B", "spindle"]
            elif fault == "circuit_anomaly":
                possible_locs = ["motor", "control_panel"]
            elif fault == "overheat_overload":
                possible_locs = ["motor", "hydraulic_system"]
            elif fault == "transmission_jam":
                possible_locs = ["gearbox", "spindle"]
            elif fault == "lubrication_deficiency":
                possible_locs = ["bearing_A", "bearing_B", "gearbox"]
            elif fault == "sensor_drift":
                possible_locs = ["control_panel"]
            else:
                possible_locs = FAULT_LOCATIONS

            confidence = row["confidence"]
            loc_confidences = {loc: confidence * (0.6 if i == 0 else 0.25 if i == 1 else 0.15)
                             for i, loc in enumerate(possible_locs[:3])}
            predicted_loc = possible_locs[0]

            locations.append(
                {
                    "predicted_location": predicted_loc,
                    "location_confidences": loc_confidences,
                }
            )

        loc_df = pd.DataFrame(locations)
        return pd.concat([fault_results, loc_df], axis=1)

    def partial_fit(self, df, fault_labels):
        if not self.is_fitted:
            return self.fit(df, fault_labels)

        X = self._extract_features(df)
        if self.feature_cols:
            X = X[self.feature_cols]

        new_labels = set(fault_labels) - set(self.classes_)
        if new_labels:
            all_labels = list(self.classes_) + list(new_labels)
            self.label_encoder.fit(all_labels)
            self.classes_ = list(self.label_encoder.classes_)

        y = self.label_encoder.transform(fault_labels)

        if self.method == "random_forest":
            self.model.n_estimators += 20
            self.model.fit(X.values, y)
        else:
            self.model.fit(X.values, y)

        return self

    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        importances = dict(zip(self.feature_cols, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def evaluate(self, df, fault_labels):
        predictions = self.predict(df)
        report = classification_report(fault_labels, predictions["predicted_fault"], output_dict=True)
        cm = confusion_matrix(fault_labels, predictions["predicted_fault"])
        return {"classification_report": report, "confusion_matrix": cm.tolist()}

    def save(self, path):
        data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_cols": self.feature_cols,
            "method": self.method,
            "is_fitted": self.is_fitted,
            "classes_": self.classes_,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.feature_cols = data["feature_cols"]
        self.method = data["method"]
        self.is_fitted = data["is_fitted"]
        self.classes_ = data["classes_"]
        return self


def generate_synthetic_fault_data(n_samples_per_fault=200):
    from data_collection.sensor_simulator import EquipmentFleet

    fleet = EquipmentFleet(count=5)
    all_data = []
    all_labels = []
    all_locations = []

    normal_data = fleet.generate_batch_readings()
    for _ in range(n_samples_per_fault // 5):
        batch = fleet.generate_batch_readings()
        all_data.append(batch)
        all_labels.extend(["normal"] * len(batch))
        all_locations.extend(["none"] * len(batch))

    for fault_type in FAULT_TYPES:
        for loc in FAULT_LOCATIONS[:2]:
            eq = list(fleet.equipments.values())[0]
            eq.inject_fault(fault_type, loc, severity=0.5)
            fault_samples = []
            for _ in range(n_samples_per_fault // 2):
                reading = eq.generate_reading()
                fault_samples.append(reading)
            fault_df = pd.DataFrame(fault_samples)
            all_data.append(fault_df)
            all_labels.extend([fault_type] * len(fault_df))
            all_locations.extend([loc] * len(fault_df))

    combined = pd.concat(all_data, ignore_index=True)
    return combined, all_labels, all_locations
