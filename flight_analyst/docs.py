"""Maps failure classes to ArduPilot wiki pages."""


# cf. https://ardupilot.org/copter/docs/ for the full wiki
WIKI_URLS = {
    "vibration": (
        "https://ardupilot.org/copter/docs/common-measuring-vibration.html"
    ),
    "vibration_damping": (
        "https://ardupilot.org/copter/docs/common-vibration-damping.html"
    ),
    "notch_filter": (
        "https://ardupilot.org/copter/docs/common-imu-notch-filtering.html"
    ),
    "throttle_notch": (
        "https://ardupilot.org/copter/docs/"
        "common-throttle-based-notch.html"
    ),
    "batch_sampling": (
        "https://ardupilot.org/copter/docs/"
        "common-imu-batchsampling.html"
    ),
    "compass_calibration": (
        "https://ardupilot.org/copter/docs/"
        "common-compass-calibration-in-mission-planner.html"
    ),
    "compass_advanced": (
        "https://ardupilot.org/copter/docs/"
        "common-compass-setup-advanced.html"
    ),
    "magnetic_interference": (
        "https://ardupilot.org/copter/docs/"
        "common-magnetic-interference.html"
    ),
    "ekf_overview": (
        "https://ardupilot.org/copter/docs/common-apm-navigation-extended-kalman-filter-overview.html"
    ),
    "ekf_failsafe": (
        "https://ardupilot.org/copter/docs/ekf-inav-failsafe.html"
    ),
    "gps_failsafe": (
        "https://ardupilot.org/copter/docs/gps-failsafe-glitch-protection.html"
    ),
    "battery_failsafe": (
        "https://ardupilot.org/copter/docs/failsafe-battery.html"
    ),
    "failsafe_overview": (
        "https://ardupilot.org/copter/docs/failsafe-landing-page.html"
    ),
    "radio_failsafe": (
        "https://ardupilot.org/copter/docs/radio-failsafe.html"
    ),
    "crash_check": (
        "https://ardupilot.org/copter/docs/crash-check.html"
    ),
    "esc_calibration": (
        "https://ardupilot.org/copter/docs/esc-calibration.html"
    ),
    "dshot": (
        "https://ardupilot.org/copter/docs/common-dshot.html"
    ),
    "motor_thrust": (
        "https://ardupilot.org/copter/docs/motor-thrust-scaling.html"
    ),
    "tuning": (
        "https://ardupilot.org/copter/docs/tuning.html"
    ),
    "prearm_checks": (
        "https://ardupilot.org/copter/docs/prearm_safety_checks.html"
    ),
    "diagnosing_logs": (
        "https://ardupilot.org/copter/docs/common-diagnosing-problems-using-logs.html"
    ),
    "log_messages": (
        "https://ardupilot.org/copter/docs/logmessages.html"
    ),
    "troubleshooting": (
        "https://ardupilot.org/copter/docs/troubleshooting.html"
    ),
}

# failure class -> list of (title, wiki_key, relevance_note)
FAILURE_DOCS = {
    "high_vibration": [
        ("Measuring Vibration", "vibration",
         "How to measure and interpret vibration levels"),
        ("Vibration Damping", "vibration_damping",
         "Practical ways to reduce vibration on the airframe"),
        ("Notch Filter Setup", "notch_filter",
         "Configure the harmonic notch filter to reject motor noise"),
    ],
    "notch_misconfigured": [
        ("IMU Notch Filtering", "notch_filter",
         "Full guide to harmonic notch filter configuration"),
        ("Throttle-Based Notch", "throttle_notch",
         "Using throttle as a frequency source for the notch filter"),
        ("IMU Batch Sampling", "batch_sampling",
         "Collect FFT data to identify the correct notch frequency"),
    ],
    "compass_error": [
        ("Compass Calibration", "compass_calibration",
         "Step-by-step compass calibration procedure"),
        ("Advanced Compass Setup", "compass_advanced",
         "Multi-compass configuration and priority settings"),
        ("Magnetic Interference", "magnetic_interference",
         "Identifying and reducing magnetic interference sources"),
    ],
    "ekf_failure": [
        ("EKF Overview", "ekf_overview",
         "How the Extended Kalman Filter works in ArduPilot"),
        ("EKF / DCM Failsafe", "ekf_failsafe",
         "Configuring the EKF failsafe and understanding its triggers"),
        ("Diagnosing with Logs", "diagnosing_logs",
         "General log-analysis techniques for EKF issues"),
    ],
    "gps_glitch": [
        ("GPS Failsafe & Glitch Protection", "gps_failsafe",
         "How ArduPilot detects and responds to GPS glitches"),
        ("EKF Overview", "ekf_overview",
         "How the EKF fuses GPS data and handles outages"),
    ],
    "battery_failsafe": [
        ("Battery Failsafe", "battery_failsafe",
         "Setting up low-voltage and consumed-mAh failsafe actions"),
        ("Failsafe Overview", "failsafe_overview",
         "Summary of all available failsafe mechanisms"),
    ],
    "radio_failsafe": [
        ("Radio Failsafe", "radio_failsafe",
         "Configuring the RC-link-loss failsafe"),
        ("Failsafe Overview", "failsafe_overview",
         "Summary of all available failsafe mechanisms"),
    ],
    "crash": [
        ("Crash Check", "crash_check",
         "How ArduPilot detects a crash and disarms the motors"),
        ("Diagnosing with Logs", "diagnosing_logs",
         "Post-crash log analysis workflow"),
        ("Troubleshooting", "troubleshooting",
         "Common failure modes and remedies"),
    ],
    "motor_imbalance": [
        ("Motor Thrust Scaling", "motor_thrust",
         "Understanding and correcting motor thrust differences"),
        ("ESC Calibration", "esc_calibration",
         "Calibrating ESCs so all motors produce equal thrust"),
        ("DShot", "dshot",
         "Using DShot for more consistent motor control"),
    ],
    "esc_desync": [
        ("DShot", "dshot",
         "DShot protocol for reliable ESC communication"),
        ("ESC Calibration", "esc_calibration",
         "Ensuring all ESCs are properly calibrated"),
        ("Troubleshooting", "troubleshooting",
         "General troubleshooting for ESC-related issues"),
    ],
    "tuning_issue": [
        ("Tuning Guide", "tuning",
         "Rate PID and filter tuning procedure"),
        ("Vibration Damping", "vibration_damping",
         "Reducing vibrations that interfere with PID performance"),
        ("Notch Filter Setup", "notch_filter",
         "Filter noise before it reaches the PID controller"),
    ],
    "prearm_failure": [
        ("Pre-arm Safety Checks", "prearm_checks",
         "Explanation of every pre-arm check and how to resolve failures"),
        ("Troubleshooting", "troubleshooting",
         "Common pre-arm failure causes and fixes"),
    ],
    "ekf_variance": [
        ("EKF Overview", "ekf_overview",
         "Understanding EKF variance and innovation metrics"),
        ("EKF / DCM Failsafe", "ekf_failsafe",
         "Variance-based failsafe thresholds and actions"),
        ("Compass Calibration", "compass_calibration",
         "Poor compass calibration is a common variance driver"),
    ],
    "power_brownout": [
        ("Battery Failsafe", "battery_failsafe",
         "Detecting and reacting to power supply issues"),
        ("Failsafe Overview", "failsafe_overview",
         "All failsafe mechanisms that can trigger on power loss"),
        ("Troubleshooting", "troubleshooting",
         "Diagnosing brownouts and voltage sags"),
    ],
    "log_quality": [
        ("Log Messages", "log_messages",
         "Reference for every log message type and field"),
        ("Diagnosing with Logs", "diagnosing_logs",
         "How to collect and inspect high-quality logs"),
    ],
}

# TODO: add Blimp/Tracker once their wikis stabilise
_VEHICLE_PATH_MAP = {
    "Copter": "copter",
    "Plane": "plane",
    "Rover": "rover",
    "Sub": "sub",
}


def _adjust_url_for_vehicle(url: str, vehicle_type: str) -> str:
    """Swap the /copter/ segment for the right vehicle path."""
    target = _VEHICLE_PATH_MAP.get(vehicle_type, "copter")
    return url.replace("/copter/", f"/{target}/")


class DocRecommender:
    """Map failure classes to ArduPilot wiki links."""

    def recommend(
        self,
        failure_class: str,
        vehicle_type: str = "Copter",
    ) -> list[dict[str, str]]:
        """Return doc recommendations for a failure class.

        URLs get rewritten so the path matches the vehicle wiki.
        Returns an empty list for unknown failure classes.
        """
        entries = FAILURE_DOCS.get(failure_class)
        if entries is None:
            return []

        results: list[dict[str, str]] = []
        for title, wiki_key, relevance_note in entries:
            base_url = WIKI_URLS[wiki_key]
            url = _adjust_url_for_vehicle(base_url, vehicle_type)
            results.append({
                "title": title,
                "url": url,
                "relevance_note": relevance_note,
            })
        return results

    @staticmethod
    def get_all_failure_classes() -> list[str]:
        """Return sorted list of known failure classes."""
        return sorted(FAILURE_DOCS.keys())
