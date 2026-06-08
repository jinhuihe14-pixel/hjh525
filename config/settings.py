import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

SENSOR_METRICS = [
    "vibration_freq",
    "temperature",
    "current",
    "voltage",
    "rotational_speed",
    "load",
    "runtime_hours",
    "noise_level",
    "pressure",
    "flow_rate",
]

EQUIPMENT_TYPES = ["cnc_machine", "stamping_press", "conveyor"]

EQUIPMENT_COUNT = 30

SAMPLING_INTERVAL_SEC = 1

WINDOW_SIZE = 60
WINDOW_STEP = 10

ANOMALY_THRESHOLD_WARNING = 2.0
ANOMALY_THRESHOLD_CRITICAL = 3.5

FAULT_TYPES = [
    "bearing_wear",
    "circuit_anomaly",
    "overheat_overload",
    "transmission_jam",
    "lubrication_deficiency",
    "sensor_drift",
]

FAULT_LOCATIONS = [
    "spindle",
    "motor",
    "gearbox",
    "bearing_A",
    "bearing_B",
    "hydraulic_system",
    "control_panel",
]

MAINTENANCE_STAFF = [
    {"id": "tech_001", "name": "张工", "skills": ["bearing_wear", "transmission_jam", "lubrication_deficiency"], "workshop": "A"},
    {"id": "tech_002", "name": "李工", "skills": ["circuit_anomaly", "sensor_drift", "control_panel"], "workshop": "A"},
    {"id": "tech_003", "name": "王工", "skills": ["overheat_overload", "hydraulic_system", "motor"], "workshop": "B"},
    {"id": "tech_004", "name": "赵工", "skills": ["bearing_wear", "transmission_jam", "gearbox"], "workshop": "B"},
    {"id": "tech_005", "name": "刘工", "skills": ["circuit_anomaly", "overheat_overload", "spindle"], "workshop": "A"},
]

WORKSHOP_LAYOUT = {
    "A": ["cnc_machine_001", "cnc_machine_002", "cnc_machine_003", "stamping_press_001", "stamping_press_002"],
    "B": ["cnc_machine_004", "cnc_machine_005", "conveyor_001", "conveyor_002", "stamping_press_003"],
}

PRIORITY_LEVELS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

RUL_WARNING_DAYS = 7
RUL_CRITICAL_DAYS = 2

DATABASE_PATH = os.path.join(DATA_DIR, "predictive_maintenance.db")
TIMESERIES_CSV = os.path.join(DATA_DIR, "timeseries_data.csv")
FAULT_RECORDS_CSV = os.path.join(DATA_DIR, "fault_records.csv")
WORKORDERS_CSV = os.path.join(DATA_DIR, "workorders.csv")
EQUIPMENT_META_CSV = os.path.join(DATA_DIR, "equipment_meta.csv")
