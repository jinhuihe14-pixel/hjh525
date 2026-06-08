import sqlite3
import pandas as pd
import os
import json
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager

from config.settings import (
    DATABASE_PATH,
    SENSOR_METRICS,
    FAULT_TYPES,
    EQUIPMENT_TYPES,
)


class TimeSeriesStore:
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _serialize_value(self, value):
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        elif isinstance(value, datetime):
            return value.isoformat()
        return value

    def _serialize_dict(self, d):
        return {k: self._serialize_value(v) for k, v in d.items()}

    def _init_db(self):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS timeseries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL,
                        equipment_type TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        health_score REAL,
                        vibration_freq REAL,
                        temperature REAL,
                        current REAL,
                        voltage REAL,
                        rotational_speed REAL,
                        load REAL,
                        runtime_hours REAL,
                        noise_level REAL,
                        pressure REAL,
                        flow_rate REAL,
                        anomaly_score REAL,
                        anomaly_level TEXT,
                        is_anomaly INTEGER DEFAULT 0
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ts_equipment 
                    ON timeseries(equipment_id, timestamp DESC)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ts_timestamp 
                    ON timeseries(timestamp DESC)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ts_anomaly 
                    ON timeseries(is_anomaly, timestamp DESC)
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fault_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL,
                        fault_type TEXT,
                        fault_location TEXT,
                        severity REAL,
                        detected_at DATETIME,
                        resolved_at DATETIME,
                        description TEXT,
                        root_cause TEXT,
                        solution TEXT,
                        parts_replaced TEXT,
                        downtime_minutes REAL,
                        cost REAL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workorders (
                        id TEXT PRIMARY KEY,
                        equipment_id TEXT NOT NULL,
                        fault_type TEXT,
                        fault_location TEXT,
                        priority TEXT,
                        status TEXT,
                        source TEXT,
                        assigned_to TEXT,
                        created_at DATETIME,
                        assigned_at DATETIME,
                        started_at DATETIME,
                        completed_at DATETIME,
                        description TEXT,
                        notes TEXT,
                        solution TEXT,
                        root_cause TEXT,
                        parts_used TEXT,
                        downtime_minutes REAL,
                        actual_downtime_minutes REAL,
                        cost REAL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS equipment_meta (
                        equipment_id TEXT PRIMARY KEY,
                        equipment_type TEXT,
                        workshop TEXT,
                        install_date DATETIME,
                        initial_health REAL,
                        current_health REAL,
                        rul_days REAL,
                        total_runtime_hours REAL,
                        fault_count INTEGER DEFAULT 0,
                        last_maintenance DATETIME,
                        next_maintenance DATETIME,
                        priority_level TEXT
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS anomaly_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL,
                        alert_level TEXT,
                        anomaly_score REAL,
                        metric TEXT,
                        timestamp DATETIME,
                        message TEXT,
                        acknowledged INTEGER DEFAULT 0
                    )
                """)

                conn.commit()

    def insert_reading(self, reading):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                columns = ", ".join(reading.keys())
                placeholders = ", ".join(["?"] * len(reading))
                values = tuple(reading.values())
                cursor.execute(
                    f"INSERT INTO timeseries ({columns}) VALUES ({placeholders})",
                    values,
                )
                conn.commit()
                return cursor.lastrowid

    def insert_batch(self, readings_df):
        if readings_df.empty:
            return 0
        with self._lock:
            with self._get_conn() as conn:
                readings_df.to_sql(
                    "timeseries", conn, if_exists="append", index=False
                )
                conn.commit()
                return len(readings_df)

    def query_equipment(self, equipment_id, start_time=None, end_time=None, limit=1000):
        query = "SELECT * FROM timeseries WHERE equipment_id = ?"
        params = [equipment_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

    def query_all_equipments_latest(self, limit_per_eq=100):
        query = """
            SELECT t.* FROM timeseries t
            INNER JOIN (
                SELECT equipment_id, MAX(timestamp) as max_ts
                FROM timeseries
                GROUP BY equipment_id
            ) latest ON t.equipment_id = latest.equipment_id
            WHERE t.timestamp >= datetime(latest.max_ts, '-' || ? || ' seconds')
            ORDER BY t.equipment_id, t.timestamp DESC
        """
        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=[limit_per_eq])

    def query_anomalies(self, level=None, start_time=None, limit=100):
        query = "SELECT * FROM timeseries WHERE is_anomaly = 1"
        params = []

        if level:
            query += " AND anomaly_level = ?"
            params.append(level)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_equipment_list(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT equipment_id, equipment_type FROM timeseries")
            return [dict(row) for row in cursor.fetchall()]

    def get_stats_summary(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM timeseries")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(DISTINCT equipment_id) as eq_count FROM timeseries")
            eq_count = cursor.fetchone()["eq_count"]

            cursor.execute("SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest FROM timeseries")
            ts_range = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) as anomaly_count FROM timeseries WHERE is_anomaly = 1")
            anomaly_count = cursor.fetchone()["anomaly_count"]

            return {
                "total_readings": total,
                "equipment_count": eq_count,
                "earliest_timestamp": ts_range["earliest"],
                "latest_timestamp": ts_range["latest"],
                "anomaly_count": anomaly_count,
            }

    def insert_fault_record(self, fault_data):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                fault_data_serialized = self._serialize_dict(fault_data)
                columns = ", ".join(fault_data_serialized.keys())
                placeholders = ", ".join(["?"] * len(fault_data_serialized))
                values = tuple(fault_data_serialized.values())
                cursor.execute(
                    f"INSERT INTO fault_records ({columns}) VALUES ({placeholders})",
                    values,
                )
                conn.commit()
                return cursor.lastrowid

    def get_fault_records(self, equipment_id=None, limit=100):
        query = "SELECT * FROM fault_records"
        params = []
        if equipment_id:
            query += " WHERE equipment_id = ?"
            params.append(equipment_id)
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def upsert_equipment_meta(self, meta):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                meta_serialized = self._serialize_dict(meta)
                columns = list(meta_serialized.keys())
                placeholders = ", ".join(["?"] * len(columns))
                update_clause = ", ".join([f"{c} = excluded.{c}" for c in columns if c != "equipment_id"])
                query = f"""
                    INSERT INTO equipment_meta ({', '.join(columns)}) 
                    VALUES ({placeholders})
                    ON CONFLICT(equipment_id) DO UPDATE SET {update_clause}
                """
                cursor.execute(query, tuple(meta_serialized.values()))
                conn.commit()

    def get_equipment_meta(self, equipment_id=None):
        with self._get_conn() as conn:
            if equipment_id:
                df = pd.read_sql_query(
                    "SELECT * FROM equipment_meta WHERE equipment_id = ?",
                    conn,
                    params=[equipment_id],
                )
                return df.iloc[0].to_dict() if not df.empty else None
            else:
                return pd.read_sql_query("SELECT * FROM equipment_meta", conn)

    def insert_workorder(self, workorder):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                workorder_serialized = self._serialize_dict(workorder)
                columns = list(workorder_serialized.keys())
                placeholders = ", ".join(["?"] * len(columns))
                cursor.execute(
                    f"INSERT INTO workorders ({', '.join(columns)}) VALUES ({placeholders})",
                    tuple(workorder_serialized.values()),
                )
                conn.commit()
                return workorder.get("id")

    def update_workorder(self, wo_id, updates):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                updates_serialized = self._serialize_dict(updates)
                set_clause = ", ".join([f"{k} = ?" for k in updates_serialized.keys()])
                query = f"UPDATE workorders SET {set_clause} WHERE id = ?"
                cursor.execute(query, tuple(list(updates_serialized.values()) + [wo_id]))
                conn.commit()
                return cursor.rowcount > 0

    def get_workorders(self, status=None, limit=100):
        query = "SELECT * FROM workorders"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def insert_alert(self, alert_data):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                alert_data_serialized = self._serialize_dict(alert_data)
                columns = ", ".join(alert_data_serialized.keys())
                placeholders = ", ".join(["?"] * len(alert_data_serialized))
                cursor.execute(
                    f"INSERT INTO anomaly_alerts ({columns}) VALUES ({placeholders})",
                    tuple(alert_data_serialized.values()),
                )
                conn.commit()
                return cursor.lastrowid

    def get_alerts(self, equipment_id=None, level=None, acknowledged=None, limit=100):
        query = "SELECT * FROM anomaly_alerts"
        params = []
        conditions = []

        if equipment_id:
            conditions.append("equipment_id = ?")
            params.append(equipment_id)
        if level:
            conditions.append("alert_level = ?")
            params.append(level)
        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def acknowledge_alert(self, alert_id):
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE anomaly_alerts SET acknowledged = 1 WHERE id = ?",
                    (alert_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
