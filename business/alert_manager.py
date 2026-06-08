import uuid
from datetime import datetime
from collections import defaultdict
import threading

from config.settings import PRIORITY_LEVELS, FAULT_TYPES


class AlertManager:
    def __init__(self, store=None):
        self.store = store
        self._subscribers = defaultdict(list)
        self._alert_history = []
        self._suppressed_equipment = set()
        self._cooldown_period = 300
        self._last_alert_time = {}
        self._lock = threading.Lock()

    def subscribe(self, alert_level, callback):
        self._subscribers[alert_level].append(callback)

    def unsubscribe(self, alert_level, callback):
        if callback in self._subscribers.get(alert_level, []):
            self._subscribers[alert_level].remove(callback)

    def _should_alert(self, equipment_id, alert_level):
        with self._lock:
            key = (equipment_id, alert_level)
            now = datetime.now()

            if equipment_id in self._suppressed_equipment:
                return False

            last_time = self._last_alert_time.get(key)
            if last_time:
                elapsed = (now - last_time).total_seconds()
                if elapsed < self._cooldown_period:
                    return False

            self._last_alert_time[key] = now
            return True

    def create_alert(self, equipment_id, alert_level, anomaly_score, metric=None, message=None):
        if not self._should_alert(equipment_id, alert_level):
            return None

        if message is None:
            level_texts = {
                "critical": "严重故障前兆，建议立即停机检查",
                "warning": "检测到异常指标，建议安排巡检",
                "info": "设备状态波动，持续关注",
            }
            message = f"设备 {equipment_id} {level_texts.get(alert_level, '状态异常')}"
            if metric:
                message += f"，异常指标：{metric}"

        alert_data = {
            "equipment_id": equipment_id,
            "alert_level": alert_level,
            "anomaly_score": anomaly_score,
            "metric": metric,
            "timestamp": datetime.now(),
            "message": message,
            "acknowledged": 0,
        }

        if self.store:
            alert_id = self.store.insert_alert(alert_data)
            alert_data["id"] = alert_id

        self._alert_history.append(alert_data)

        self._notify_subscribers(alert_level, alert_data)

        return alert_data

    def _notify_subscribers(self, alert_level, alert_data):
        for callback in self._subscribers.get(alert_level, []):
            try:
                callback(alert_data)
            except Exception as e:
                print(f"Alert callback error: {e}")

        for callback in self._subscribers.get("all", []):
            try:
                callback(alert_data)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def process_anomaly_result(self, equipment_id, anomaly_result):
        level = anomaly_result.get("anomaly_level", "normal")
        if level == "normal":
            return None

        priority_map = {
            "critical": "critical",
            "warning": "warning",
        }
        alert_level = priority_map.get(level, "info")

        score = anomaly_result.get("anomaly_score", 0)
        metric = anomaly_result.get("max_anomaly_metric")

        return self.create_alert(
            equipment_id=equipment_id,
            alert_level=alert_level,
            anomaly_score=score,
            metric=metric,
        )

    def suppress_equipment(self, equipment_id):
        with self._lock:
            self._suppressed_equipment.add(equipment_id)

    def unsuppress_equipment(self, equipment_id):
        with self._lock:
            self._suppressed_equipment.discard(equipment_id)

    def get_recent_alerts(self, limit=50, level=None):
        alerts = self._alert_history.copy()
        if level:
            alerts = [a for a in alerts if a["alert_level"] == level]
        return alerts[-limit:]

    def get_alert_stats(self):
        stats = defaultdict(int)
        for alert in self._alert_history:
            stats[alert["alert_level"]] += 1
            stats["total"] += 1
        return dict(stats)

    def acknowledge_alert(self, alert_id):
        if self.store:
            return self.store.acknowledge_alert(alert_id)
        for alert in self._alert_history:
            if alert.get("id") == alert_id:
                alert["acknowledged"] = 1
                return True
        return False
