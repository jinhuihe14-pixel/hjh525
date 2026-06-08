import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.sensor_simulator import EquipmentFleet, EquipmentSensorSimulator
from data_collection.preprocessor import DataPreprocessor
from models.anomaly_detection import (
    ZScoreAnomalyDetector,
    IsolationForestAnomalyDetector,
    EnsembleAnomalyDetector,
)
from models.fault_classifier import FaultClassifier, generate_synthetic_fault_data
from models.rul_predictor import RULPredictor
from models.maintenance_scheduler import MaintenanceScheduler
from business.alert_manager import AlertManager
from business.workorder_manager import WorkOrderManager
from business.health_report import HealthReportGenerator


def test_sensor_simulator():
    print("=" * 50)
    print("测试 1: 传感器模拟器")
    print("=" * 50)
    fleet = EquipmentFleet(count=5)
    print(f"✓ 创建了 {len(fleet.equipments)} 台设备")

    eq = list(fleet.equipments.values())[0]
    reading = eq.generate_reading()
    print(f"✓ 单台设备读数: {reading['equipment_id']}")
    print(f"  健康度: {reading['health_score']:.4f}")
    print(f"  振动频率: {reading['vibration_freq']:.2f} Hz")
    print(f"  温度: {reading['temperature']:.2f} °C")

    batch = fleet.generate_batch_readings()
    print(f"✓ 批量读数: {len(batch)} 条记录")

    eq_id, fault_type, location, severity = fleet.inject_random_fault()
    print(f"✓ 注入故障: {eq_id} - {fault_type} at {location}")

    print()


def test_preprocessor():
    print("=" * 50)
    print("测试 2: 数据预处理")
    print("=" * 50)
    fleet = EquipmentFleet(count=3)
    all_data = []
    for _ in range(100):
        batch = fleet.generate_batch_readings()
        all_data.append(batch)
    df = pd.concat(all_data, ignore_index=True)

    preprocessor = DataPreprocessor(method="standard")
    preprocessor.fit_scaler(df)
    print(f"✓ 拟合数据: {len(df)} 条记录")

    denoised = preprocessor.denoise_all(df, method="kalman")
    print(f"✓ 卡尔曼滤波降噪完成")

    normalized = preprocessor.normalize(denoised)
    print(f"✓ 标准化完成")

    windows, info = preprocessor.sliding_windows(normalized, window_size=30, step=10)
    print(f"✓ 滑动窗口: {len(windows)} 个窗口, 形状: {windows.shape}")

    features = preprocessor.extract_features(windows)
    print(f"✓ 特征提取: {features.shape[1]} 维特征")

    print()


def test_anomaly_detection():
    print("=" * 50)
    print("测试 3: 异常检测模型")
    print("=" * 50)
    fleet = EquipmentFleet(count=3)
    normal_data = []
    for _ in range(200):
        batch = fleet.generate_batch_readings()
        normal_data.append(batch)
    df = pd.concat(normal_data, ignore_index=True)

    preprocessor = DataPreprocessor()
    preprocessor.fit_scaler(df)
    normalized = preprocessor.normalize(df)

    zscore_detector = ZScoreAnomalyDetector()
    zscore_detector.fit(normalized)
    print(f"✓ Z-Score 检测器训练完成")

    iforest_detector = IsolationForestAnomalyDetector(contamination=0.05)
    iforest_detector.fit(normalized)
    print(f"✓ 孤立森林检测器训练完成")

    ensemble = EnsembleAnomalyDetector()
    ensemble.fit(normalized)
    print(f"✓ 集成检测器训练完成")

    eq = list(fleet.equipments.values())[0]
    eq.inject_fault("bearing_wear", "bearing_A", severity=0.6)
    fault_data = []
    for _ in range(50):
        reading = eq.generate_reading()
        fault_data.append(reading)
    fault_df = pd.DataFrame(fault_data)
    fault_normalized = preprocessor.normalize(fault_df)

    results = ensemble.detect(fault_normalized)
    anomaly_count = results["is_anomaly"].sum()
    print(f"✓ 故障检测: {anomaly_count}/{len(results)} 个异常点")
    print(f"  最高异常分数: {results['anomaly_score'].max():.4f}")
    print(f"  严重异常: {(results['anomaly_level'] == 'critical').sum()} 个")
    print(f"  警告: {(results['anomaly_level'] == 'warning').sum()} 个")

    print()


def test_fault_classifier():
    print("=" * 50)
    print("测试 4: 故障分类模型")
    print("=" * 50)
    df, labels, locations = generate_synthetic_fault_data(n_samples_per_fault=150)

    preprocessor = DataPreprocessor()
    preprocessor.fit_scaler(df)
    normalized = preprocessor.normalize(df)

    classifier = FaultClassifier(method="random_forest")
    classifier.fit(normalized, labels)
    print(f"✓ 故障分类器训练完成")
    print(f"  类别数: {len(classifier.classes_)}")

    predictions = classifier.predict(normalized)
    correct = (predictions["predicted_fault"] == pd.Series(labels)).sum()
    accuracy = correct / len(labels)
    print(f"✓ 训练集准确率: {accuracy:.2%}")

    result = classifier.predict_single(normalized.iloc[0].to_dict())
    print(f"✓ 单样本预测: {result['predicted_fault']}, 置信度: {result['confidence']:.2%}")

    with_loc = classifier.predict_with_location(normalized[:5])
    print(f"✓ 带位置预测完成，样例: {with_loc.iloc[0]['predicted_location']}")

    importances = classifier.get_feature_importance()
    top_features = list(importances.items())[:5]
    print(f"✓ Top 5 特征重要性:")
    for feat, imp in top_features:
        print(f"    {feat}: {imp:.4f}")

    print()


def test_rul_predictor():
    print("=" * 50)
    print("测试 5: RUL 剩余使用寿命预测")
    print("=" * 50)
    predictor = RULPredictor()

    from datetime import datetime, timedelta

    start_time = datetime.now() - timedelta(days=60)
    equipment_id = "test_eq_001"

    health_values = []
    for i in range(60 * 24):
        t = i / (60 * 24)
        base_health = 0.95 - 0.6 * t
        noise = np.random.normal(0, 0.01)
        health = max(0.2, base_health + noise)
        health_values.append(health)
        ts = start_time + timedelta(hours=i)
        predictor.add_health_reading(equipment_id, health, ts)

    print(f"✓ 添加了 {len(health_values)} 个历史健康点")

    result = predictor.predict_rul(equipment_id, failure_threshold=0.3, method="linear")
    print(f"✓ 线性预测 RUL: {result['rul_days']:.1f} 天, 置信度: {result['confidence']:.2%}")

    result_exp = predictor.predict_rul(equipment_id, failure_threshold=0.3, method="exponential")
    print(f"✓ 指数预测 RUL: {result_exp['rul_days']:.1f} 天")

    result_poly = predictor.predict_rul(equipment_id, failure_threshold=0.3, method="polynomial")
    print(f"✓ 多项式预测 RUL: {result_poly['rul_days']:.1f} 天")

    ensemble_result = predictor.predict_rul_ensemble(equipment_id, failure_threshold=0.3)
    print(f"✓ 集成预测 RUL: {ensemble_result['rul_days']:.1f} 天")
    print(f"  状态: {ensemble_result['status']}")
    print(f"  置信度: {ensemble_result['confidence']:.2%}")
    print(f"  建议维保窗口: {ensemble_result['recommended_maintenance_window']['earliest']} ~ {ensemble_result['recommended_maintenance_window']['latest']}")

    print()


def test_maintenance_scheduler():
    print("=" * 50)
    print("测试 6: 运维智能调度")
    print("=" * 50)
    scheduler = MaintenanceScheduler()

    print(f"✓ 运维人员: {len(scheduler.staff)} 人")
    print(f"  空闲: {len(scheduler.get_available_staff())} 人")

    orders = [
        {"id": "WO-001", "equipment_id": "cnc_machine_001", "fault_type": "bearing_wear", "priority": "high", "created_at": datetime.now()},
        {"id": "WO-002", "equipment_id": "cnc_machine_002", "fault_type": "circuit_anomaly", "priority": "critical", "created_at": datetime.now()},
        {"id": "WO-003", "equipment_id": "stamping_press_001", "fault_type": "overheat_overload", "priority": "medium", "created_at": datetime.now()},
    ]
    scheduler.add_work_orders_batch(orders)
    print(f"✓ 添加了 {len(orders)} 个工单")

    summary = scheduler.get_queue_summary()
    print(f"  待处理: {summary['pending_count']}")
    print(f"  按优先级: {summary['by_priority']}")

    assignments = scheduler.schedule_optimal()
    print(f"✓ 智能派单: 分配了 {len(assignments)} 个工单")
    for a in assignments:
        print(f"    {a['order_id']} -> {a['staff_name']} (匹配度: {a['score']:.2%})")

    summary = scheduler.get_queue_summary()
    print(f"  已分配: {summary['assigned_count']}")

    scheduler.complete_order(assignments[0]["order_id"], solution="更换轴承", parts_used=["bearing_6205"])
    print(f"✓ 完成工单 {assignments[0]['order_id']}")

    summary = scheduler.get_queue_summary()
    print(f"  已完成: {summary['completed_count']}")

    print()


def test_alert_workorder():
    print("=" * 50)
    print("测试 7: 预警与工单管理")
    print("=" * 50)
    alert_mgr = AlertManager()
    wo_mgr = WorkOrderManager()

    alert_count = 0

    def on_critical(alert):
        nonlocal alert_count
        alert_count += 1

    alert_mgr.subscribe("critical", on_critical)
    print("✓ 订阅严重预警通知")

    alert1 = alert_mgr.create_alert(
        equipment_id="cnc_001",
        alert_level="critical",
        anomaly_score=4.2,
        metric="vibration_freq",
    )
    print(f"✓ 产生严重预警: {alert1['message']}")

    alert2 = alert_mgr.create_alert(
        equipment_id="cnc_002",
        alert_level="warning",
        anomaly_score=2.5,
        metric="temperature",
    )
    print(f"✓ 产生警告: {alert2['message']}")

    stats = alert_mgr.get_alert_stats()
    print(f"  预警统计: {stats}")

    wo = wo_mgr.create_from_alert(alert1, fault_type="bearing_wear")
    print(f"✓ 从预警创建工单: {wo['id']}, 优先级: {wo['priority']}")

    wo2 = wo_mgr.create_workorder(
        equipment_id="cnc_003",
        fault_type="overheat_overload",
        priority="medium",
        description="电机温度偏高",
    )
    print(f"✓ 手动创建工单: {wo2['id']}")

    wo_stats = wo_mgr.get_stats()
    print(f"  工单统计: {wo_stats['by_status']}")

    assignments = wo_mgr.schedule_pending()
    print(f"✓ 自动派单: {len(assignments)} 个工单")

    wo_mgr.complete_workorder(
        assignments[0]["order_id"],
        solution="更换轴承并加注润滑脂",
        parts_used=["bearing_6205", "grease_001"],
        notes="设备运行恢复正常",
        downtime_minutes=90,
        cost=1500.0,
    )
    print(f"✓ 完成工单 {assignments[0]['order_id']}")

    print()


def test_health_report():
    print("=" * 50)
    print("测试 8: 健康报表与决策支持")
    print("=" * 50)

    fleet = EquipmentFleet(count=10)
    all_data = {}
    for eq_id, eq in fleet.equipments.items():
        readings = []
        for _ in range(200):
            readings.append(eq.generate_reading())
        all_data[eq_id] = pd.DataFrame(readings)

    report_gen = HealthReportGenerator()
    fleet_report = report_gen.generate_fleet_report(all_data)
    print(f"✓ 生成机群健康报告")
    print(f"  设备总数: {fleet_report['total_equipment']}")
    print(f"  整体健康度: {fleet_report['fleet_overall_health']:.2%}")
    print(f"  等级分布: {fleet_report['health_level_distribution']}")
    print(f"  趋势分布: {fleet_report['trend_distribution']}")

    workorders = [
        {"id": "WO-1", "equipment_id": "cnc_machine_001", "fault_type": "bearing_wear", "status": "completed", "downtime_minutes": 120, "cost": 2000},
        {"id": "WO-2", "equipment_id": "cnc_machine_002", "fault_type": "circuit_anomaly", "status": "completed", "downtime_minutes": 60, "cost": 800},
        {"id": "WO-3", "equipment_id": "stamping_press_001", "fault_type": "overheat_overload", "status": "completed", "downtime_minutes": 45, "cost": 500},
        {"id": "WO-4", "equipment_id": "cnc_machine_001", "fault_type": "transmission_jam", "status": "in_progress", "downtime_minutes": 0, "cost": 0},
    ]

    cost_report = report_gen.generate_maintenance_cost_report(workorders)
    print(f"\n✓ 生成维保成本报告")
    print(f"  总工单: {cost_report['total_workorders']}")
    print(f"  总成本: ¥{cost_report['total_cost']:,.2f}")
    print(f"  总停机: {cost_report['total_downtime_hours']:.1f} 小时")
    print(f"  平均成本: ¥{cost_report['average_repair_cost']:,.2f}")

    decision_report = report_gen.generate_decision_support_report(fleet_report, cost_report)
    print(f"\n✓ 生成决策支持报告")
    print(f"  整体风险: {decision_report['risk_assessment']['overall_risk']}")
    print(f"  故障概率: {decision_report['risk_assessment']['breakdown_probability']:.1%}")
    print(f"  建议数: {len(decision_report['recommendations'])} 条")
    for i, rec in enumerate(decision_report["recommendations"][:2], 1):
        print(f"    [{i}] {rec['description']}")

    print("\n✓ 格式化报表演示:")
    print(report_gen.format_report_text(fleet_report, "fleet")[:500] + "...")

    print()


def run_all_tests():
    print("\n" + "=" * 60)
    print("  设备预测性维护系统 - 模块测试")
    print("=" * 60 + "\n")

    import pandas as pd
    global pd

    try:
        test_sensor_simulator()
        test_preprocessor()
        test_anomaly_detection()
        test_fault_classifier()
        test_rul_predictor()
        test_maintenance_scheduler()
        test_alert_workorder()
        test_health_report()

        print("=" * 60)
        print("  所有模块测试通过！ ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime

    run_all_tests()
