"""Predefined SITL failure scenarios for dataset generation."""

SCENARIOS = {
    "motor_failure_hover": {
        "label": "motor_failure",
        "description": "Single motor failure during hover",
        "setup": {"target_alt": 30, "hover_time": 10, "mode": "LOITER"},
        "fault": {"type": "motor_failure", "motor": 1, "efficiency": 0.0},
        "observe_time": 15,
    },
    "motor_degradation_hover": {
        "label": "motor_imbalance",
        "description": "Motor at 50% efficiency",
        "setup": {"target_alt": 30, "hover_time": 10, "mode": "LOITER"},
        "fault": {"type": "motor_failure", "motor": 2, "efficiency": 0.5},
        "observe_time": 30,
    },
    "gps_glitch_loiter": {
        "label": "gps_glitch",
        "description": "GPS position jump during loiter",
        "setup": {"target_alt": 20, "hover_time": 5, "mode": "LOITER"},
        "fault": {"type": "gps_glitch", "lat_offset": 0.001, "lon_offset": 0.001},
        "observe_time": 20,
    },
    "gps_loss_loiter": {
        "label": "gps_glitch",
        "description": "Complete GPS loss",
        "setup": {"target_alt": 20, "hover_time": 5, "mode": "LOITER"},
        "fault": {"type": "gps_loss"},
        "observe_time": 30,
    },
    "compass_interference": {
        "label": "compass_interference",
        "description": "Large compass offset",
        "setup": {"target_alt": 20, "hover_time": 10, "mode": "LOITER"},
        "fault": {"type": "compass_interference", "offset": 300},
        "observe_time": 20,
    },
    "battery_failsafe": {
        "label": "power_issue",
        "description": "Battery voltage drop",
        "setup": {
            "target_alt": 30,
            "hover_time": 5,
            "mode": "LOITER",
            "params": {
                "BATT_LOW_VOLT": 10.5,
                "BATT_CRT_VOLT": 9.8,
                "BATT_FS_LOW_ACT": 2,
                "BATT_FS_CRT_ACT": 1,
            },
        },
        "fault": {"type": "battery_drop", "voltage": 10.0},
        "observe_time": 30,
    },
    "high_vibration": {
        "label": "vibration_high",
        "description": "Severe vibration causing EKF issues",
        "setup": {"target_alt": 20, "hover_time": 10, "mode": "LOITER"},
        "fault": {"type": "vibration", "freq": 150, "noise": 10},
        "observe_time": 30,
    },
    "rc_failsafe": {
        "label": "rc_failsafe",
        "description": "RC transmitter loss",
        "setup": {
            "target_alt": 30,
            "hover_time": 5,
            "mode": "LOITER",
            "params": {"FS_THR_ENABLE": 1},
        },
        "fault": {"type": "rc_failsafe"},
        "observe_time": 30,
    },
    "ekf_divergence": {
        "label": "ekf_failure",
        "description": "IMU bias causing EKF failure",
        "setup": {"target_alt": 20, "hover_time": 10, "mode": "LOITER"},
        "fault": {"type": "imu_bias", "accel_bias": 5.0},
        "observe_time": 20,
    },
    "normal_flight": {
        "label": "normal",
        "description": "Clean flight (negative sample)",
        "setup": {"target_alt": 30, "hover_time": 30, "mode": "LOITER"},
        "fault": None,
        "observe_time": 0,
    },
}

_REQUIRED_KEYS = {"label", "description", "setup", "fault", "observe_time"}
_REQUIRED_SETUP_KEYS = {"target_alt", "hover_time", "mode"}


def get_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    return list(SCENARIOS.keys())


def validate_scenario(scenario: dict) -> bool:
    """Check that *scenario* has all required keys. Returns True/False."""
    if not isinstance(scenario, dict):
        return False

    if not _REQUIRED_KEYS.issubset(scenario.keys()):
        return False

    setup = scenario.get("setup")
    if not isinstance(setup, dict):
        return False
    if not _REQUIRED_SETUP_KEYS.issubset(setup.keys()):
        return False

    if not isinstance(scenario["observe_time"], (int, float)):
        return False

    fault = scenario.get("fault")
    if fault is not None:
        if not isinstance(fault, dict):
            return False
        if "type" not in fault:
            return False

    return True
