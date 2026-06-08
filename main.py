import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.system import PredictiveMaintenanceSystem
from business.health_report import HealthReportGenerator


def demo_normal_operation():
    print("=" * 70)
    print("  设备预测性维护与智能调度系统 - 演示模式")
    print("=" * 70)

    system = PredictiveMaintenanceSystem()

    print("\n[Step 1] 正在初始化系统和模型...")
    system.initialize_models(training_duration_sec=10)
    print("✓ 模型初始化完成")

    print("\n[Step 2] 启动数据采集与实时分析...")
    system.start()
    print("✓ 系统运行中...")

    time.sleep(2)

    print("\n[Step 3] 查看系统状态...")
    status = system.get_system_status()
    print(f"  - 设备数量: {status['equipment_count']}")
    print(f"  - 已采集数据点: {status['data_collection']['total_samples']}")
    print(f"  - 数据存储记录数: {status['storage']['total_readings']}")

    print("\n[Step 4] 查看设备列表...")
    equipments = system.list_equipment()
    print(f"  共 {len(equipments)} 台设备，展示前 5 台:")
    for eq in equipments[:5]:
        print(f"    - {eq['equipment_id']}: 健康度={eq['current_health']:.2%}, RUL={eq['rul_days']}天")

    print("\n[Step 5] 注入测试故障 (模拟设备劣化)...")
    result = system.inject_test_fault()
    if result:
        eq_id, fault_type, location, severity = result
        print(f"  ✓ 已向设备 {eq_id} 注入 {fault_type} 故障")
        print(f"    位置: {location}, 严重度: {severity:.2f}")

    print("\n[Step 6] 等待故障征兆显现...")
    for i in range(5, 0, -1):
        print(f"  等待 {i} 秒...", end="\r")
        time.sleep(1)
    print("  等待完成      ")

    print("\n[Step 7] 查看故障设备详情...")
    if result:
        eq_id = result[0]
        detail = system.get_equipment_detail(eq_id)
        print(f"  设备: {eq_id}")
        print(f"  整体健康度: {detail['health']['overall_health']:.2%}")
        print(f"  健康等级: {detail['health']['health_level']}")
        print(f"  退化趋势: {detail['health']['trend']}")
        print(f"  异常次数: {detail['health']['anomaly_count']}")
        print(f"  建议: {detail['health']['recommendation']}")
        print(f"  剩余使用寿命: {detail['rul'].get('rul_days', 'N/A')}天")
        print(f"  RUL置信度: {detail['rul'].get('confidence', 0):.1%}")

    print("\n[Step 8] 查看预警信息...")
    alerts = system.alert_manager.get_recent_alerts(limit=10)
    print(f"  共产生 {len(alerts)} 条预警:")
    for alert in alerts[:5]:
        print(f"    [{alert['alert_level']}] {alert['message']}")

    print("\n[Step 9] 查看工单状态...")
    wo_stats = system.workorder_manager.get_stats()
    print(f"  总工单: {wo_stats['by_status'].get('total', 0)}")
    print(f"  待处理: {wo_stats['by_status'].get('pending', 0)}")
    print(f"  已分配: {wo_stats['by_status'].get('assigned', 0)}")
    print(f"  已完成: {wo_stats['by_status'].get('completed', 0)}")

    print("\n[Step 10] 生成设备健康报告...")
    reports = system.generate_report(report_type="full")
    report_gen = HealthReportGenerator()

    print("\n" + "=" * 70)
    print("  机群健康报告")
    print("=" * 70)
    fleet_report = reports["fleet_health"]
    print(f"  设备总数: {fleet_report['total_equipment']}")
    print(f"  整体健康度: {fleet_report['fleet_overall_health']:.2%}")
    print(f"  健康等级分布:")
    for level, count in fleet_report["health_level_distribution"].items():
        print(f"    {level}: {count}台")
    print(f"\n  需重点关注设备 (Top 3):")
    for eq in fleet_report["needs_attention"][:3]:
        print(f"    - {eq['equipment_id']}: {eq['overall_health']:.2%}")

    print("\n" + "=" * 70)
    print("  决策支持报告")
    print("=" * 70)
    decision = reports["decision_support"]
    print(f"  整体风险等级: {decision['risk_assessment']['overall_risk']}")
    print(f"  故障概率估计: {decision['risk_assessment']['breakdown_probability']:.1%}")
    print(f"\n  优化建议:")
    for i, rec in enumerate(decision["recommendations"][:3], 1):
        print(f"    [{i}] {rec['description']}")
        print(f"        优先级: {rec['priority']}")

    print("\n" + "=" * 70)
    print("  系统运行总结")
    print("=" * 70)
    final_status = system.get_system_status()
    print(f"  运行状态: {'运行中' if final_status['is_running'] else '已停止'}")
    print(f"  采集数据点: {final_status['data_collection']['total_samples']}")
    print(f"  存储记录数: {final_status['storage']['total_readings']}")
    print(f"  异常记录数: {final_status['storage']['anomaly_count']}")
    print(f"  生成预警数: {sum(final_status['alerts'].values())}")
    print(f"  总工单数: {final_status['workorders']['by_status'].get('total', 0)}")

    print("\n[Step 11] 停止系统...")
    system.stop()
    print("✓ 系统已停止")

    print("\n" + "=" * 70)
    print("  演示完成！")
    print("=" * 70)
    return system


def demo_report_only():
    print("=" * 70)
    print("  设备健康报表生成演示")
    print("=" * 70)

    system = PredictiveMaintenanceSystem()
    system.initialize_models(training_duration_sec=5)

    print("\n生成模拟历史数据...")
    all_data = []
    for _ in range(100):
        batch = system.data_collector.fleet.generate_batch_readings()
        all_data.append(batch)
    combined = pd.concat(all_data, ignore_index=True)
    system.store.insert_batch(combined)

    print("\n生成机群健康报告...")
    reports = system.generate_report(report_type="full")

    report_gen = HealthReportGenerator()

    print("\n" + report_gen.format_report_text(reports["fleet_health"], "fleet"))
    print("\n" + report_gen.format_report_text(reports["maintenance_cost"], "cost"))
    print("\n" + report_gen.format_report_text(reports["decision_support"], "decision"))

    return system


def main():
    parser = argparse.ArgumentParser(
        description="设备预测性维护与智能调度系统"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "report", "interactive"],
        default="demo",
        help="运行模式: demo(完整演示), report(报表演示), interactive(交互模式)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="演示运行时长(秒)",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demo_normal_operation()
    elif args.mode == "report":
        demo_report_only()
    elif args.mode == "interactive":
        print("交互模式 - 输入命令操作系统")
        print("可用命令: start, stop, status, inject, report, equipment, quit")
        system = PredictiveMaintenanceSystem()
        system.initialize_models(training_duration_sec=5)

        while True:
            try:
                cmd = input("\n> ").strip().lower()
                if cmd == "quit" or cmd == "exit":
                    system.stop()
                    break
                elif cmd == "start":
                    system.start()
                elif cmd == "stop":
                    system.stop()
                elif cmd == "status":
                    status = system.get_system_status()
                    print(f"运行状态: {'运行中' if status['is_running'] else '已停止'}")
                    print(f"设备数: {status['equipment_count']}")
                    print(f"采集样本: {status['data_collection']['total_samples']}")
                    print(f"预警数: {sum(status['alerts'].values())}")
                elif cmd == "inject":
                    system.inject_test_fault()
                elif cmd == "report":
                    reports = system.generate_report()
                    report_gen = HealthReportGenerator()
                    print(report_gen.format_report_text(reports["fleet_health"], "fleet"))
                elif cmd == "equipment":
                    eqs = system.list_equipment()
                    for eq in eqs[:10]:
                        print(f"  {eq['equipment_id']}: 健康={eq['current_health']:.2%}, RUL={eq['rul_days']}天")
                else:
                    print("未知命令，可用: start, stop, status, inject, report, equipment, quit")
            except KeyboardInterrupt:
                system.stop()
                break
            except Exception as e:
                print(f"错误: {e}")


if __name__ == "__main__":
    import pandas as pd

    main()
