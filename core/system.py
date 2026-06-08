import threading
import time
from datetime import datetime
from collections import deque, defaultdict
import pandas as pd

from config.settings import (
    SENSOR_METRICS,
    MODEL_DIR,
    WINDOW_SIZE,
    ANOMALY_THRESHOLD_CRITICAL,
)
from data_collection.data_collector import DataCollector
from data_collection.preprocessor import DataPreprocessor
from storage.timeseries_store import TimeSeriesStore
from models.anomaly_detection import EnsembleAnomalyDetector
from models.fault_classifier import FaultClassifier
from models.rul_predictor import RULPredictor
from models.maintenance_scheduler import MaintenanceScheduler
from business.alert_manager import AlertManager
from business.workorder_manager import WorkOrderManager
from business.health_report import HealthReportGenerator


class PredictiveMaintenanceSystem:
    def __init__(self):
        self.store = TimeSeriesStore()
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor(method="standard")
        self.anomaly_detector = EnsembleAnomalyDetector()
        self.fault_classifier = FaultClassifier(method="random_forest")
        self.rul_predictor = RULPredictor()
        self.scheduler = MaintenanceScheduler()
        self.alert_manager = AlertManager(store=self.store)
        self.workorder_manager = WorkOrderManager(store=self.store, scheduler=self.scheduler)
        self.report_generator = HealthReportGenerator(store=self.store)

        self.is_running = False
        self._processing_thread = None
        self._analysis_thread = None
        self._lock = threading.Lock()

        self._recent_readings = defaultdict(lambda: deque(maxlen=WINDOW_SIZE * 2))
        self._equipment_health = {}
        self._model_initialized = False

        self._setup_callbacks()

    def _setup_callbacks(self):
        def on_alert_critical(alert):
            print(f"[CRITICAL ALERT] {alert['message']}")
            self._auto_create_workorder(alert, priority="critical")

        def on_alert_warning(alert):
            print(f"[WARNING] {alert['message']}")

        self.alert_manager.subscribe("critical", on_alert_critical)
        self.alert_manager.subscribe("warning", on_alert_warning)

    def initialize_models(self, training_duration_sec=60):
        print("[System] Initializing models with baseline data...")

        baseline_data = []
        for _ in range(training_duration_sec):
            batch = self.data_collector.fleet.generate_batch_readings()
            baseline_data.append(batch)
            time.sleep(0.01)

        all_data = pd.concat(baseline_data, ignore_index=True)

        self.preprocessor.fit_scaler(all_data)
        normalized_data = self.preprocessor.normalize(all_data)
        self.anomaly_detector.fit(normalized_data)

        self._init_fault_classifier()
        self._init_rul_predictor(all_data)

        for eq_id, eq in self.data_collector.fleet.equipments.items():
            self.rul_predictor.add_health_reading(eq_id, eq.health)

        self._model_initialized = True
        print("[System] Models initialized successfully")
        return True

    def _init_fault_classifier(self):
        from models.fault_classifier import generate_synthetic_fault_data

        fault_df, fault_labels, _ = generate_synthetic_fault_data(n_samples_per_fault=100)

        normalized_fault_df = fault_df.copy()
        metric_cols = [m for m in SENSOR_METRICS if m in normalized_fault_df.columns]
        for col in metric_cols:
            if col in self.preprocessor._metric_stats:
                stats = self.preprocessor._metric_stats[col]
                mean = stats.get("mean", 0)
                std = stats.get("std", 1)
                if std > 0:
                    normalized_fault_df[col] = (normalized_fault_df[col] - mean) / std

        self.fault_classifier.fit(normalized_fault_df, fault_labels)

    def _init_rul_predictor(self, initial_data):
        equipment_groups = initial_data.groupby("equipment_id")
        for eq_id, group in equipment_groups:
            for _, row in group.iterrows():
                self.rul_predictor.add_health_reading(
                    eq_id,
                    row.get("health_score", 0.8),
                    row.get("timestamp"),
                )

    def _processing_loop(self):
        while self.is_running:
            try:
                batch = self.data_collector.get_all_available()
                if not batch.empty:
                    self._process_batch(batch)
            except Exception as e:
                print(f"[Processing Error] {e}")

            time.sleep(0.5)

    def _analysis_loop(self):
        while self.is_running:
            try:
                self._periodic_analysis()
            except Exception as e:
                print(f"[Analysis Error] {e}")

            time.sleep(30)

    def _process_batch(self, batch_df):
        normalized = self.preprocessor.normalize(batch_df)

        anomaly_results = self.anomaly_detector.detect(normalized)

        enhanced_df = batch_df.copy()
        enhanced_df["anomaly_score"] = anomaly_results["anomaly_score"].values
        enhanced_df["anomaly_level"] = anomaly_results["anomaly_level"].values
        enhanced_df["is_anomaly"] = anomaly_results["is_anomaly"].values.astype(int)

        self.store.insert_batch(enhanced_df)

        for i, row in enhanced_df.iterrows():
            eq_id = row["equipment_id"]
            self._recent_readings[eq_id].append(row.to_dict())
            self._equipment_health[eq_id] = row.get("health_score", 0.8)

            if row["is_anomaly"] == 1:
                anomaly_info = anomaly_results.iloc[i].to_dict()
                self.alert_manager.process_anomaly_result(eq_id, anomaly_info)

                if row["anomaly_score"] >= ANOMALY_THRESHOLD_CRITICAL:
                    reading_dict = normalized.iloc[i].to_dict()
                    fault_result = self.fault_classifier.predict_single(reading_dict)
                    self._handle_critical_anomaly(eq_id, row, fault_result)

            self.rul_predictor.add_health_reading(
                eq_id, row.get("health_score", 0.8), row.get("timestamp")
            )

    def _handle_critical_anomaly(self, equipment_id, reading, fault_result):
        predicted_fault = fault_result.get("predicted_fault", "unknown")
        confidence = fault_result.get("confidence", 0)

        fault_info = {
            "equipment_id": equipment_id,
            "fault_type": predicted_fault,
            "severity": reading["anomaly_score"] / 5,
            "detected_at": datetime.now(),
            "description": f"检测到{predicted_fault}故障前兆，置信度{confidence:.1%}",
        }

        try:
            self.store.insert_fault_record(fault_info)
        except Exception:
            pass

    def _auto_create_workorder(self, alert, priority="high"):
        eq_id = alert["equipment_id"]

        readings = list(self._recent_readings.get(eq_id, []))
        fault_type = None
        if readings:
            try:
                latest_reading = readings[-1]
                normalized = self.preprocessor.normalize(pd.DataFrame([latest_reading]))
                fault_result = self.fault_classifier.predict_single(normalized.iloc[0].to_dict())
                fault_type = fault_result.get("predicted_fault")
            except Exception:
                pass

        workorder = self.workorder_manager.create_from_alert(
            alert, fault_type=fault_type, priority=priority
        )

        if workorder:
            assignments = self.workorder_manager.schedule_pending()
            if assignments:
                print(f"[Dispatch] Workorder {workorder['id']} assigned automatically")

        return workorder

    def _periodic_analysis(self):
        rul_results = self.rul_predictor.predict_all()

        for eq_id, result in rul_results.items():
            meta = {
                "equipment_id": eq_id,
                "current_health": self._equipment_health.get(eq_id, 0.8),
                "rul_days": result.get("rul_days"),
                "priority_level": result.get("status", "normal"),
            }
            try:
                self.store.upsert_equipment_meta(meta)
            except Exception:
                pass

        pending_count = self.workorder_manager.get_pending_count()
        if pending_count > 0:
            self.workorder_manager.schedule_pending()

    def start(self):
        if self.is_running:
            print("[System] Already running")
            return

        print("[System] Starting Predictive Maintenance System...")

        if not self._model_initialized:
            self.initialize_models(training_duration_sec=10)

        self.data_collector.start()

        self.is_running = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._processing_thread.start()
        self._analysis_thread.start()

        print("[System] System started successfully")
        return True

    def stop(self):
        if not self.is_running:
            return

        print("[System] Stopping system...")
        self.is_running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=2)
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2)

        self.data_collector.stop()
        print("[System] System stopped")

    def get_system_status(self):
        collector_stats = self.data_collector.get_stats()
        store_stats = self.store.get_stats_summary()
        wo_stats = self.workorder_manager.get_stats()
        alert_stats = self.alert_manager.get_alert_stats()

        return {
            "is_running": self.is_running,
            "models_initialized": self._model_initialized,
            "data_collection": collector_stats,
            "storage": store_stats,
            "workorders": wo_stats,
            "alerts": alert_stats,
            "equipment_count": len(self._equipment_health),
            "timestamp": datetime.now(),
        }

    def inject_test_fault(self, equipment_id=None):
        result = self.data_collector.inject_fault(equipment_id)
        if result:
            eq_id, fault_type, location, severity = result
            print(f"[Test] Fault injected: {eq_id} - {fault_type} at {location}, severity={severity:.2f}")
            return result
        return None

    def generate_report(self, report_type="full"):
        if report_type == "fleet" or report_type == "full":
            fleet_report = self.report_generator.generate_fleet_report()
        else:
            fleet_report = None

        if report_type == "cost" or report_type == "full":
            cost_report = self.report_generator.generate_maintenance_cost_report()
        else:
            cost_report = None

        if report_type == "decision" or report_type == "full":
            decision_report = self.report_generator.generate_decision_support_report(
                fleet_report, cost_report
            )
        else:
            decision_report = None

        return {
            "fleet_health": fleet_report,
            "maintenance_cost": cost_report,
            "decision_support": decision_report,
        }

    def get_equipment_detail(self, equipment_id):
        df = self.store.query_equipment(equipment_id, limit=200)
        health_info = self.report_generator.calculate_equipment_health_score(equipment_id, df)
        rul_info = self.rul_predictor.predict_rul_ensemble(equipment_id)
        meta = self.store.get_equipment_meta(equipment_id)

        return {
            "health": health_info,
            "rul": rul_info,
            "meta": meta,
            "recent_data_points": len(df),
        }

    def list_equipment(self):
        eq_list = []
        for eq_id, health in self._equipment_health.items():
            rul_info = self.rul_predictor.predict_rul_ensemble(eq_id)
            eq_list.append(
                {
                    "equipment_id": eq_id,
                    "current_health": health,
                    "rul_days": rul_info.get("rul_days"),
                    "rul_status": rul_info.get("status"),
                }
            )
        return eq_list
