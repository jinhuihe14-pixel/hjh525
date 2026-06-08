import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import heapq
from collections import defaultdict

from config.settings import (
    MAINTENANCE_STAFF,
    PRIORITY_LEVELS,
    WORKSHOP_LAYOUT,
    FAULT_TYPES,
)


class MaintenanceScheduler:
    def __init__(self, staff_list=None):
        self.staff = staff_list or MAINTENANCE_STAFF.copy()
        self.staff_status = {}
        self.work_order_queue = []
        self.assigned_orders = []
        self.completed_orders = []
        self._init_staff_status()

    def _init_staff_status(self):
        for s in self.staff:
            self.staff_status[s["id"]] = {
                **s,
                "status": "idle",
                "current_order": None,
                "completed_today": 0,
                "workload_score": 0,
            }

    def _calculate_skill_match(self, staff_id, fault_type, location=None):
        staff = self.staff_status.get(staff_id)
        if not staff:
            return 0

        skills = staff.get("skills", [])
        if fault_type in skills:
            base_score = 1.0
        else:
            related = 0
            for skill in skills:
                if any(k in fault_type for k in skill.split("_")):
                    related += 1
            base_score = min(0.5, related * 0.2)

        if location and location in skills:
            base_score += 0.3

        return min(1.0, base_score)

    def _calculate_proximity(self, staff_workshop, equipment_id):
        for workshop, eqs in WORKSHOP_LAYOUT.items():
            if equipment_id in eqs:
                return 1.0 if workshop == staff_workshop else 0.5
        return 0.7

    def _estimate_repair_time(self, fault_type, priority):
        base_times = {
            "bearing_wear": 120,
            "circuit_anomaly": 90,
            "overheat_overload": 60,
            "transmission_jam": 150,
            "lubrication_deficiency": 30,
            "sensor_drift": 45,
            "normal": 20,
        }
        base = base_times.get(fault_type, 60)

        priority_multipliers = {
            "critical": 1.5,
            "high": 1.2,
            "medium": 1.0,
            "low": 0.8,
        }
        return base * priority_multipliers.get(priority, 1.0)

    def add_work_order(self, work_order):
        priority_val = PRIORITY_LEVELS.get(work_order.get("priority", "medium"), 2)
        heapq.heappush(
            self.work_order_queue,
            (-priority_val, work_order.get("created_at", datetime.now()), work_order),
        )
        return len(self.work_order_queue)

    def add_work_orders_batch(self, orders):
        for order in orders:
            self.add_work_order(order)
        return len(self.work_order_queue)

    def get_available_staff(self):
        return [s for s in self.staff_status.values() if s["status"] == "idle"]

    def assign_order(self, work_order, staff_id):
        staff = self.staff_status.get(staff_id)
        if not staff or staff["status"] != "idle":
            return False

        staff["status"] = "busy"
        staff["current_order"] = work_order["id"]
        staff["workload_score"] += PRIORITY_LEVELS.get(work_order.get("priority", "medium"), 2)

        work_order["assigned_to"] = staff_id
        work_order["assigned_at"] = datetime.now()
        work_order["status"] = "assigned"
        work_order["estimated_duration_min"] = self._estimate_repair_time(
            work_order.get("fault_type", "normal"),
            work_order.get("priority", "medium"),
        )

        self.assigned_orders.append(work_order)
        return True

    def complete_order(self, order_id, notes="", parts_used=None, solution=""):
        for i, order in enumerate(self.assigned_orders):
            if order["id"] == order_id:
                order["status"] = "completed"
                order["completed_at"] = datetime.now()
                order["notes"] = notes
                order["parts_used"] = parts_used or []
                order["solution"] = solution

                staff_id = order.get("assigned_to")
                if staff_id and staff_id in self.staff_status:
                    self.staff_status[staff_id]["status"] = "idle"
                    self.staff_status[staff_id]["current_order"] = None
                    self.staff_status[staff_id]["completed_today"] += 1

                self.assigned_orders.pop(i)
                self.completed_orders.append(order)
                return True
        return False

    def schedule_optimal(self):
        assignments = []
        available_staff = self.get_available_staff()
        pending_orders = []

        temp_queue = []
        while self.work_order_queue:
            priority, created_at, order = heapq.heappop(self.work_order_queue)
            pending_orders.append(order)
            temp_queue.append((priority, created_at, order))

        for priority, created_at, order in temp_queue:
            heapq.heappush(self.work_order_queue, (priority, created_at, order))

        if not available_staff or not pending_orders:
            return assignments

        assignment_scores = {}
        for order in pending_orders:
            for staff in available_staff:
                skill_score = self._calculate_skill_match(
                    staff["id"], order.get("fault_type", "normal"), order.get("location")
                )
                proximity = self._calculate_proximity(staff["workshop"], order["equipment_id"])
                workload = 1 - (staff["workload_score"] / 10)
                total_score = 0.5 * skill_score + 0.3 * proximity + 0.2 * workload
                assignment_scores[(order["id"], staff["id"])] = total_score

        assigned_staff = set()
        assigned_orders = set()

        sorted_assignments = sorted(
            assignment_scores.items(), key=lambda x: x[1], reverse=True
        )

        for (order_id, staff_id), score in sorted_assignments:
            if staff_id in assigned_staff or order_id in assigned_orders:
                continue

            order = next((o for o in pending_orders if o["id"] == order_id), None)
            if not order:
                continue

            if self.assign_order(order, staff_id):
                assigned_staff.add(staff_id)
                assigned_orders.add(order_id)
                assignments.append(
                    {
                        "order_id": order_id,
                        "staff_id": staff_id,
                        "staff_name": self.staff_status[staff_id]["name"],
                        "score": round(score, 4),
                    }
                )

        return assignments

    def get_queue_summary(self):
        by_priority = defaultdict(int)
        for _, _, order in self.work_order_queue:
            by_priority[order.get("priority", "medium")] += 1

        return {
            "pending_count": len(self.work_order_queue),
            "by_priority": dict(by_priority),
            "assigned_count": len(self.assigned_orders),
            "completed_count": len(self.completed_orders),
            "staff_count": len(self.staff_status),
            "idle_staff_count": len(self.get_available_staff()),
        }

    def get_staff_status(self, staff_id=None):
        if staff_id:
            return self.staff_status.get(staff_id)
        return list(self.staff_status.values())

    def get_pending_orders(self):
        orders = []
        temp_queue = []
        while self.work_order_queue:
            priority, created_at, order = heapq.heappop(self.work_order_queue)
            orders.append(order)
            temp_queue.append((priority, created_at, order))
        for item in temp_queue:
            heapq.heappush(self.work_order_queue, item)
        return orders

    def rebalance_workload(self):
        if len(self.assigned_orders) == 0:
            return []

        workloads = {
            sid: s["workload_score"] for sid, s in self.staff_status.items()
        }
        max_wl = max(workloads.values()) if workloads else 0
        min_wl = min(workloads.values()) if workloads else 0

        reassignments = []
        if max_wl - min_wl > 3 and len(self.assigned_orders) > 1:
            pass

        return reassignments
