import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict


class HealthReportGenerator:
    def __init__(self, store=None):
        self.store = store

    def calculate_equipment_health_score(self, equipment_id, df=None):
        if df is None and self.store:
            df = self.store.query_equipment(equipment_id, limit=500)

        if df is None or df.empty:
            return {
                "overall_health": 0.8,
                "components": {},
                "trend": "stable",
                "recommendation": "数据不足",
            }

        health_scores = df["health_score"].values if "health_score" in df.columns else None
        anomaly_count = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
        anomaly_ratio = anomaly_count / len(df) if len(df) > 0 else 0

        if health_scores is not None and len(health_scores) > 0:
            current_health = float(health_scores[-1])
            avg_health = float(np.mean(health_scores))
            min_health = float(np.min(health_scores))

            if len(health_scores) > 10:
                recent = health_scores[-10:]
                earlier = health_scores[:10] if len(health_scores) >= 20 else health_scores[: len(health_scores) // 2]
                trend_val = np.mean(recent) - np.mean(earlier)
                if trend_val > 0.02:
                    trend = "improving"
                elif trend_val < -0.02:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            current_health = 1 - anomaly_ratio * 2
            avg_health = current_health
            min_health = current_health
            trend = "stable"

        components = {
            "vibration_health": 0.9,
            "thermal_health": 0.85,
            "electrical_health": 0.95,
            "mechanical_health": 0.88,
        }

        if "vibration_freq" in df.columns:
            vib_std = df["vibration_freq"].std()
            vib_mean = df["vibration_freq"].mean()
            vib_cv = vib_std / vib_mean if vib_mean > 0 else 0
            components["vibration_health"] = max(0.3, 1 - vib_cv * 5)

        if "temperature" in df.columns:
            temp_max = df["temperature"].max()
            temp_mean = df["temperature"].mean()
            components["thermal_health"] = max(0.3, 1 - (temp_max - temp_mean) / temp_mean * 2)

        if "current" in df.columns and "voltage" in df.columns:
            curr_std = df["current"].std()
            curr_mean = df["current"].mean()
            volt_std = df["voltage"].std()
            volt_mean = df["voltage"].mean()
            elec_stability = 1 - ((curr_std / curr_mean if curr_mean > 0 else 0) + (volt_std / volt_mean if volt_mean > 0 else 0)) / 2
            components["electrical_health"] = max(0.3, elec_stability)

        overall = sum(components.values()) / len(components)
        overall = overall * 0.7 + current_health * 0.3
        overall = max(0.1, min(1.0, overall))

        if overall >= 0.9:
            level = "excellent"
            recommendation = "设备状态良好，按计划维保即可"
        elif overall >= 0.75:
            level = "good"
            recommendation = "设备状态正常，建议加强日常巡检"
        elif overall >= 0.6:
            level = "fair"
            recommendation = "设备存在轻微退化，建议安排预防性维护"
        elif overall >= 0.4:
            level = "poor"
            recommendation = "设备退化明显，建议尽快安排维保"
        else:
            level = "critical"
            recommendation = "设备状态危急，建议立即停机检修"

        return {
            "equipment_id": equipment_id,
            "overall_health": round(float(overall), 4),
            "health_level": level,
            "components": {k: round(float(v), 4) for k, v in components.items()},
            "current_health_score": round(float(current_health), 4) if current_health else None,
            "average_health": round(float(avg_health), 4) if avg_health else None,
            "min_health": round(float(min_health), 4) if min_health else None,
            "trend": trend,
            "anomaly_count": int(anomaly_count),
            "anomaly_ratio": round(float(anomaly_ratio), 4),
            "recommendation": recommendation,
            "data_points": len(df),
        }

    def generate_fleet_report(self, equipment_data=None):
        if equipment_data is None and self.store:
            eq_list = self.store.get_equipment_list()
            equipment_data = {}
            for eq in eq_list:
                eq_id = eq["equipment_id"]
                df = self.store.query_equipment(eq_id, limit=200)
                equipment_data[eq_id] = df

        health_scores = {}
        for eq_id, df in equipment_data.items():
            health_scores[eq_id] = self.calculate_equipment_health_score(eq_id, df)

        overall_scores = [h["overall_health"] for h in health_scores.values()]
        levels = [h["health_level"] for h in health_scores.values()]
        trends = [h["trend"] for h in health_scores.values()]

        level_counts = defaultdict(int)
        for l in levels:
            level_counts[l] += 1

        trend_counts = defaultdict(int)
        for t in trends:
            trend_counts[t] += 1

        fleet_health = np.mean(overall_scores) if overall_scores else 0.8

        sorted_by_health = sorted(health_scores.values(), key=lambda x: x["overall_health"])

        report = {
            "generated_at": datetime.now(),
            "total_equipment": len(health_scores),
            "fleet_overall_health": round(float(fleet_health), 4),
            "health_level_distribution": dict(level_counts),
            "trend_distribution": dict(trend_counts),
            "best_performing": sorted_by_health[-5:][::-1] if len(sorted_by_health) >= 5 else sorted_by_health[::-1],
            "needs_attention": sorted_by_health[:5],
            "equipment_details": health_scores,
            "summary": {
                "excellent_count": level_counts.get("excellent", 0),
                "good_count": level_counts.get("good", 0),
                "fair_count": level_counts.get("fair", 0),
                "poor_count": level_counts.get("poor", 0),
                "critical_count": level_counts.get("critical", 0),
            },
        }

        return report

    def generate_maintenance_cost_report(self, workorders=None):
        if workorders is None and self.store:
            workorders = self.store.get_workorders(limit=1000).to_dict("records")

        if not workorders:
            return {
                "total_cost": 0,
                "total_downtime_hours": 0,
                "cost_by_equipment": {},
                "cost_by_fault_type": {},
                "average_repair_cost": 0,
            }

        total_cost = sum(wo.get("cost", 0) for wo in workorders)
        total_downtime = sum(wo.get("downtime_minutes", 0) for wo in workorders)

        cost_by_eq = defaultdict(float)
        cost_by_fault = defaultdict(float)
        downtime_by_eq = defaultdict(float)

        for wo in workorders:
            eq_id = wo.get("equipment_id", "unknown")
            fault = wo.get("fault_type", "unknown")
            cost = wo.get("cost", 0)
            downtime = wo.get("downtime_minutes", 0)

            cost_by_eq[eq_id] += cost
            cost_by_fault[fault] += cost
            downtime_by_eq[eq_id] += downtime

        completed = [wo for wo in workorders if wo.get("status") == "completed"]
        avg_cost = total_cost / len(completed) if completed else 0

        return {
            "total_workorders": len(workorders),
            "completed_count": len(completed),
            "total_cost": round(float(total_cost), 2),
            "total_downtime_hours": round(float(total_downtime / 60), 2),
            "average_repair_cost": round(float(avg_cost), 2),
            "average_downtime_minutes": round(float(total_downtime / len(completed)) if completed else 0, 2),
            "cost_by_equipment": {k: round(float(v), 2) for k, v in sorted(cost_by_eq.items(), key=lambda x: x[1], reverse=True)},
            "cost_by_fault_type": {k: round(float(v), 2) for k, v in sorted(cost_by_fault.items(), key=lambda x: x[1], reverse=True)},
            "downtime_by_equipment": {k: round(float(v / 60), 2) for k, v in sorted(downtime_by_eq.items(), key=lambda x: x[1], reverse=True)},
        }

    def generate_decision_support_report(self, health_report=None, cost_report=None):
        if health_report is None:
            health_report = self.generate_fleet_report()
        if cost_report is None:
            cost_report = self.generate_maintenance_cost_report()

        recommendations = []

        poor_equipment = [
            h for h in health_report.get("equipment_details", {}).values()
            if h.get("health_level") in ["poor", "critical"]
        ]
        if poor_equipment:
            recommendations.append(
                {
                    "type": "urgent_maintenance",
                    "priority": "high",
                    "equipment_count": len(poor_equipment),
                    "description": f"{len(poor_equipment)}台设备状态较差，建议立即安排维保",
                    "estimated_benefit": "减少突发故障风险，避免产线停摆",
                }
            )

        degrading = [
            h for h in health_report.get("equipment_details", {}).values()
            if h.get("trend") == "degrading"
        ]
        if degrading:
            recommendations.append(
                {
                    "type": "preventive_maintenance",
                    "priority": "medium",
                    "equipment_count": len(degrading),
                    "description": f"{len(degrading)}台设备呈退化趋势，建议安排预防性维护",
                    "estimated_benefit": "提前处理隐患，降低严重故障概率",
                }
            )

        cost_by_fault = cost_report.get("cost_by_fault_type", {})
        if cost_by_fault:
            top_fault = max(cost_by_fault, key=cost_by_fault.get)
            recommendations.append(
                {
                    "type": "spare_parts",
                    "priority": "medium",
                    "description": f"{top_fault}类故障维修成本最高，建议增加备品备件库存",
                    "associated_cost": cost_by_fault[top_fault],
                }
            )

        fleet_health = health_report.get("fleet_overall_health", 0.8)
        if fleet_health < 0.6:
            recommendations.append(
                {
                    "type": "equipment_replacement",
                    "priority": "low",
                    "description": "设备整体健康度较低，建议制定设备更新换代计划",
                    "estimated_cycle": "1-3年",
                }
            )

        return {
            "generated_at": datetime.now(),
            "fleet_health_score": health_report.get("fleet_overall_health", 0),
            "total_maintenance_cost": cost_report.get("total_cost", 0),
            "total_downtime_hours": cost_report.get("total_downtime_hours", 0),
            "recommendations": recommendations,
            "risk_assessment": {
                "overall_risk": "low" if fleet_health > 0.8 else "medium" if fleet_health > 0.6 else "high",
                "breakdown_probability": round(max(0, 1 - fleet_health), 4),
                "cost_risk": "low" if cost_report.get("total_cost", 0) < 10000 else "medium",
            },
        }

    def format_report_text(self, report, report_type="fleet"):
        lines = []
        lines.append("=" * 60)
        lines.append(f"  设备健康报告 - {report.get('generated_at', datetime.now())}")
        lines.append("=" * 60)

        if report_type == "fleet":
            lines.append(f"  设备总数: {report.get('total_equipment', 0)}")
            lines.append(f"  整体健康度: {report.get('fleet_overall_health', 0):.2%}")
            lines.append("")
            lines.append("  健康等级分布:")
            for level, count in report.get("health_level_distribution", {}).items():
                lines.append(f"    {level}: {count}")
            lines.append("")
            lines.append("  需关注设备 (Top 5):")
            for eq in report.get("needs_attention", []):
                lines.append(
                    f"    {eq.get('equipment_id', 'N/A')}: "
                    f"{eq.get('overall_health', 0):.2%} - {eq.get('recommendation', '')}"
                )

        elif report_type == "cost":
            lines.append(f"  总工单数量: {report.get('total_workorders', 0)}")
            lines.append(f"  总维保成本: ¥{report.get('total_cost', 0):,.2f}")
            lines.append(f"  总停机时间: {report.get('total_downtime_hours', 0):.1f}小时")
            lines.append(f"  平均维修成本: ¥{report.get('average_repair_cost', 0):,.2f}")
            lines.append("")
            lines.append("  按故障类型成本分布:")
            for fault, cost in report.get("cost_by_fault_type", {}).items():
                lines.append(f"    {fault}: ¥{cost:,.2f}")

        elif report_type == "decision":
            lines.append(f"  机群健康评分: {report.get('fleet_health_score', 0):.2%}")
            lines.append(f"  总维保成本: ¥{report.get('total_maintenance_cost', 0):,.2f}")
            lines.append(f"  总停机时间: {report.get('total_downtime_hours', 0):.1f}小时")
            lines.append(f"  整体风险等级: {report.get('risk_assessment', {}).get('overall_risk', 'unknown')}")
            lines.append("")
            lines.append("  决策建议:")
            for i, rec in enumerate(report.get("recommendations", []), 1):
                lines.append(f"    [{i}] {rec.get('description', '')}")
                lines.append(f"        优先级: {rec.get('priority', 'N/A')}")

        lines.append("=" * 60)
        return "\n".join(lines)
