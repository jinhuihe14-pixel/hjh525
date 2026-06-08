import uuid
import json
from datetime import datetime
from collections import defaultdict
import threading

from config.settings import PRIORITY_LEVELS
from models.maintenance_scheduler import MaintenanceScheduler


class WorkOrderManager:
    def __init__(self, store=None, scheduler=None):
        self.store = store
        self.scheduler = scheduler or MaintenanceScheduler()
        self._work_orders = {}
        self._lock = threading.Lock()
        self._order_counter = 0

    def create_workorder(
        self,
        equipment_id,
        fault_type=None,
        priority="medium",
        description="",
        source="alert",
        location=None,
    ):
        with self._lock:
            self._order_counter += 1
            wo_id = f"WO-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:04d}"

            workorder = {
                "id": wo_id,
                "equipment_id": equipment_id,
                "fault_type": fault_type,
                "fault_location": location,
                "priority": priority,
                "status": "pending",
                "source": source,
                "assigned_to": None,
                "created_at": datetime.now(),
                "started_at": None,
                "completed_at": None,
                "description": description,
                "notes": "",
                "solution": "",
                "parts_used": [],
                "downtime_minutes": 0,
                "cost": 0.0,
            }

            self._work_orders[wo_id] = workorder

            if self.store:
                self.store.insert_workorder(workorder)

            self.scheduler.add_work_order(workorder)

            return workorder

    def create_from_alert(self, alert_data, fault_type=None, priority=None):
        if priority is None:
            level = alert_data.get("alert_level", "warning")
            priority_map = {
                "critical": "critical",
                "warning": "high",
                "info": "medium",
            }
            priority = priority_map.get(level, "medium")

        description = alert_data.get("message", "")
        location = alert_data.get("metric", "")

        return self.create_workorder(
            equipment_id=alert_data["equipment_id"],
            fault_type=fault_type,
            priority=priority,
            description=description,
            source="alert",
            location=location,
        )

    def assign_workorder(self, wo_id, staff_id):
        with self._lock:
            if wo_id not in self._work_orders:
                return False

            wo = self._work_orders[wo_id]
            wo["assigned_to"] = staff_id
            wo["status"] = "assigned"
            wo["assigned_at"] = datetime.now()

            if self.store:
                self.store.update_workorder(wo_id, {
                    "assigned_to": staff_id,
                    "status": "assigned",
                })

            return True

    def start_workorder(self, wo_id):
        with self._lock:
            if wo_id not in self._work_orders:
                return False

            wo = self._work_orders[wo_id]
            wo["status"] = "in_progress"
            wo["started_at"] = datetime.now()

            if self.store:
                self.store.update_workorder(wo_id, {"status": "in_progress"})

            return True

    def complete_workorder(
        self,
        wo_id,
        solution="",
        parts_used=None,
        notes="",
        root_cause="",
        downtime_minutes=0,
        cost=0.0,
    ):
        with self._lock:
            if wo_id not in self._work_orders:
                return False

            wo = self._work_orders[wo_id]
            wo["status"] = "completed"
            wo["completed_at"] = datetime.now()
            wo["solution"] = solution
            wo["parts_used"] = parts_used or []
            wo["notes"] = notes
            wo["root_cause"] = root_cause
            wo["downtime_minutes"] = downtime_minutes
            wo["cost"] = cost

            if wo.get("started_at") and wo["completed_at"]:
                actual_downtime = (wo["completed_at"] - wo["started_at"]).total_seconds() / 60
                wo["actual_downtime_minutes"] = actual_downtime

            if self.store:
                self.store.update_workorder(
                    wo_id,
                    {
                        "status": "completed",
                        "solution": solution,
                        "notes": notes,
                        "downtime_minutes": downtime_minutes,
                        "cost": cost,
                    },
                )

            self.scheduler.complete_order(
                wo_id, notes=notes, parts_used=parts_used, solution=solution
            )

            return wo

    def get_workorder(self, wo_id):
        return self._work_orders.get(wo_id)

    def list_workorders(self, status=None, priority=None, equipment_id=None, limit=100):
        orders = list(self._work_orders.values())

        if status:
            orders = [o for o in orders if o["status"] == status]
        if priority:
            orders = [o for o in orders if o["priority"] == priority]
        if equipment_id:
            orders = [o for o in orders if o["equipment_id"] == equipment_id]

        orders.sort(key=lambda x: x["created_at"], reverse=True)
        return orders[:limit]

    def get_stats(self):
        stats = defaultdict(int)
        by_priority = defaultdict(int)
        total_downtime = 0
        total_cost = 0.0

        for wo in self._work_orders.values():
            stats[wo["status"]] += 1
            stats["total"] += 1
            by_priority[wo["priority"]] += 1
            total_downtime += wo.get("downtime_minutes", 0)
            total_cost += wo.get("cost", 0)

        return {
            "by_status": dict(stats),
            "by_priority": dict(by_priority),
            "total_downtime_minutes": total_downtime,
            "total_cost": round(total_cost, 2),
        }

    def schedule_pending(self):
        assignments = self.scheduler.schedule_optimal()
        for assignment in assignments:
            self.assign_workorder(assignment["order_id"], assignment["staff_id"])
        return assignments

    def get_pending_count(self):
        return sum(1 for wo in self._work_orders.values() if wo["status"] == "pending")
