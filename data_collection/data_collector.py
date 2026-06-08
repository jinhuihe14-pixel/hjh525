import time
import threading
import queue
import pandas as pd
from datetime import datetime
import os

from config.settings import SAMPLING_INTERVAL_SEC, TIMESERIES_CSV
from data_collection.sensor_simulator import EquipmentFleet
from data_collection.preprocessor import DataPreprocessor


class DataCollector:
    def __init__(self, fleet=None, interval=SAMPLING_INTERVAL_SEC, buffer_size=1000):
        self.fleet = fleet or EquipmentFleet()
        self.interval = interval
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        self.is_running = False
        self._thread = None
        self._callbacks = []
        self.preprocessor = None
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self._total_samples = 0
        self._dropped_samples = 0
        self._network_available = True
        self._offline_buffer = []

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def set_network_status(self, available):
        self._network_available = available
        if available and self._offline_buffer:
            self._flush_offline_buffer()

    def _flush_offline_buffer(self):
        with self.buffer_lock:
            buffered = self._offline_buffer.copy()
            self._offline_buffer = []
        for row in buffered:
            self._push_to_queue(row)

    def _push_to_queue(self, row):
        try:
            self.data_queue.put_nowait(row)
            self._total_samples += 1
        except queue.Full:
            self._dropped_samples += 1
            self._offline_buffer.append(row)

    def _collect_loop(self):
        while self.is_running:
            start_time = time.time()
            try:
                timestamp = datetime.now()
                readings = self.fleet.generate_batch_readings(timestamp)
                for _, row in readings.iterrows():
                    row_dict = row.to_dict()
                    if self._network_available:
                        self._push_to_queue(row_dict)
                    else:
                        with self.buffer_lock:
                            self._offline_buffer.append(row_dict)

                    for cb in self._callbacks:
                        try:
                            cb(row_dict)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Data collection error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        print(f"[DataCollector] Started with {len(self.fleet.equipments)} equipments")

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)
        print(f"[DataCollector] Stopped. Total: {self._total_samples}, Dropped: {self._dropped_samples}")

    def get_latest_batch(self, count=100):
        batch = []
        for _ in range(min(count, self.data_queue.qsize())):
            try:
                batch.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return pd.DataFrame(batch) if batch else pd.DataFrame()

    def get_all_available(self):
        return self.get_latest_batch(count=self.data_queue.qsize())

    def get_stats(self):
        return {
            "total_samples": self._total_samples,
            "dropped_samples": self._dropped_samples,
            "queue_size": self.data_queue.qsize(),
            "offline_buffer_size": len(self._offline_buffer),
            "network_available": self._network_available,
            "equipment_count": len(self.fleet.equipments),
        }

    def save_to_csv(self, df, filepath=TIMESERIES_CSV):
        mode = "a" if os.path.exists(filepath) else "w"
        header = mode == "w"
        df.to_csv(filepath, mode=mode, header=header, index=False)

    def inject_fault(self, equipment_id=None, fault_type=None, location=None, severity=None):
        if equipment_id is None:
            return self.fleet.inject_random_fault()
        eq = self.fleet.get_equipment(equipment_id)
        if eq:
            import random
            from config.settings import FAULT_TYPES, FAULT_LOCATIONS
            ft = fault_type or random.choice(FAULT_TYPES)
            loc = location or random.choice(FAULT_LOCATIONS)
            sev = severity or random.uniform(0.2, 0.7)
            eq.inject_fault(ft, loc, sev)
            return equipment_id, ft, loc, sev
        return None
