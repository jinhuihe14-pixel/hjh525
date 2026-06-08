import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from config.settings import (
    SENSOR_METRICS,
    EQUIPMENT_TYPES,
    EQUIPMENT_COUNT,
    FAULT_TYPES,
    FAULT_LOCATIONS,
)


class EquipmentSensorSimulator:
    def __init__(self, equipment_id, equipment_type, initial_health=1.0):
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type
        self.health = initial_health
        self.runtime_hours = random.uniform(0, 5000)
        self._baselines = self._generate_baselines()
        self._fault_state = None
        self._fault_progression = 0.0

    def _generate_baselines(self):
        baselines = {}
        type_factors = {
            "cnc_machine": {"vibration_freq": 1.0, "temperature": 1.1, "current": 1.2, "rotational_speed": 1.0, "load": 0.8},
            "stamping_press": {"vibration_freq": 1.5, "temperature": 0.9, "current": 1.5, "rotational_speed": 0.5, "load": 1.5},
            "conveyor": {"vibration_freq": 0.7, "temperature": 0.8, "current": 0.8, "rotational_speed": 0.6, "load": 0.6},
        }
        factors = type_factors.get(self.equipment_type, {})

        baselines["vibration_freq"] = random.uniform(20, 60) * factors.get("vibration_freq", 1.0)
        baselines["temperature"] = random.uniform(35, 55) * factors.get("temperature", 1.0)
        baselines["current"] = random.uniform(8, 20) * factors.get("current", 1.0)
        baselines["voltage"] = random.uniform(210, 230)
        baselines["rotational_speed"] = random.uniform(500, 3000) * factors.get("rotational_speed", 1.0)
        baselines["load"] = random.uniform(40, 80) * factors.get("load", 1.0)
        baselines["noise_level"] = random.uniform(60, 85)
        baselines["pressure"] = random.uniform(0.4, 0.8)
        baselines["flow_rate"] = random.uniform(20, 50)
        return baselines

    def inject_fault(self, fault_type, location, severity=0.3):
        self._fault_state = {
            "type": fault_type,
            "location": location,
            "severity": severity,
        }
        self._fault_progression = 0.0

    def _apply_fault_effects(self, readings):
        if self._fault_state is None:
            return readings

        self._fault_progression = min(1.0, self._fault_progression + 0.001)
        severity = self._fault_state["severity"] * self._fault_progression
        fault_type = self._fault_state["type"]
        location = self._fault_state["location"]

        if fault_type == "bearing_wear":
            readings["vibration_freq"] *= (1 + severity * 2.5)
            readings["noise_level"] *= (1 + severity * 0.8)
            readings["temperature"] *= (1 + severity * 0.3)
            if "bearing" in location.lower():
                readings["vibration_freq"] *= (1 + severity * 1.5)

        elif fault_type == "circuit_anomaly":
            readings["voltage"] *= (1 + random.uniform(-severity * 0.3, severity * 0.3))
            readings["current"] *= (1 + severity * 1.5)
            readings["temperature"] *= (1 + severity * 0.4)

        elif fault_type == "overheat_overload":
            readings["temperature"] *= (1 + severity * 2.0)
            readings["current"] *= (1 + severity * 1.2)
            readings["load"] *= (1 + severity * 0.8)
            readings["vibration_freq"] *= (1 + severity * 0.5)

        elif fault_type == "transmission_jam":
            readings["rotational_speed"] *= (1 - severity * 0.7)
            readings["vibration_freq"] *= (1 + severity * 1.8)
            readings["current"] *= (1 + severity * 1.3)
            readings["load"] *= (1 + severity * 0.9)

        elif fault_type == "lubrication_deficiency":
            readings["vibration_freq"] *= (1 + severity * 1.5)
            readings["temperature"] *= (1 + severity * 0.8)
            readings["noise_level"] *= (1 + severity * 1.0)

        elif fault_type == "sensor_drift":
            drift_metric = random.choice(SENSOR_METRICS[:5])
            readings[drift_metric] *= (1 + severity * random.choice([-1.5, 1.5]))

        return readings

    def generate_reading(self, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        readings = {}
        for metric in SENSOR_METRICS:
            if metric == "runtime_hours":
                self.runtime_hours += 1 / 3600
                readings[metric] = round(self.runtime_hours, 4)
                continue

            base = self._baselines.get(metric, 50.0)
            noise = np.random.normal(0, base * 0.02)
            cycle = base * 0.05 * np.sin(self.runtime_hours * 0.1)
            readings[metric] = round(base + noise + cycle, 4)

        readings = self._apply_fault_effects(readings)

        self.health = max(0.1, self.health - 0.00001 * (1 - self.health) * 10)
        if self._fault_state:
            self.health = max(0.1, self.health - 0.0001 * self._fault_state["severity"])

        return {
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "timestamp": timestamp,
            "health_score": round(self.health, 4),
            **readings,
        }


class EquipmentFleet:
    def __init__(self, count=EQUIPMENT_COUNT):
        self.equipments = {}
        for i in range(1, count + 1):
            eq_type = random.choice(EQUIPMENT_TYPES)
            eq_id = f"{eq_type}_{i:03d}"
            health = random.uniform(0.75, 0.99)
            self.equipments[eq_id] = EquipmentSensorSimulator(eq_id, eq_type, health)

    def generate_batch_readings(self, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        readings = []
        for eq in self.equipments.values():
            readings.append(eq.generate_reading(timestamp))
        return pd.DataFrame(readings)

    def get_equipment(self, equipment_id):
        return self.equipments.get(equipment_id)

    def inject_random_fault(self):
        eq = random.choice(list(self.equipments.values()))
        fault_type = random.choice(FAULT_TYPES)
        location = random.choice(FAULT_LOCATIONS)
        severity = random.uniform(0.2, 0.7)
        eq.inject_fault(fault_type, location, severity)
        return eq.equipment_id, fault_type, location, severity

    def get_all_ids(self):
        return list(self.equipments.keys())
