"""Pre-flight parameter audit."""

from typing import Any


# stock default PIDs — if all match, the user never tuned
_STOCK_PIDS = {
    "ATC_RAT_RLL_P": 0.135,
    "ATC_RAT_PIT_P": 0.135,
    "ATC_RAT_YAW_P": 0.18,
}

_WIKI_BASE = "https://ardupilot.org/copter/docs"


def _warning(
    severity: str,
    param: str,
    value: Any,
    issue: str,
    recommendation: str,
    wiki_url: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "param": param,
        "value": value,
        "issue": issue,
        "recommendation": recommendation,
        "wiki_url": wiki_url,
    }


def check_tuning(params: dict[str, float]) -> list[dict[str, Any]]:
    """Warn when roll/pitch/yaw PIDs are still at stock defaults."""
    warnings: list[dict[str, Any]] = []
    stock_count = sum(
        1 for p, v in _STOCK_PIDS.items() if params.get(p) == v
    )
    if stock_count == len(_STOCK_PIDS):
        for param, default in _STOCK_PIDS.items():
            warnings.append(
                _warning(
                    severity="high",
                    param=param,
                    value=params.get(param, default),
                    issue="PID is at stock default — vehicle was never tuned",
                    recommendation=(
                        "Run AutoTune or manually adjust rate PIDs for your "
                        "frame before flying aggressively"
                    ),
                    wiki_url=f"{_WIKI_BASE}/tuning.html",
                )
            )
    return warnings


def check_notch_filter(params: dict[str, float]) -> list[dict[str, Any]]:
    if params.get("INS_HNTCH_ENABLE", 1) == 0:
        return [
            _warning(
                severity="medium",
                param="INS_HNTCH_ENABLE",
                value=0,
                issue="Harmonic notch filter is disabled",
                recommendation=(
                    "Enable and configure the harmonic notch filter to "
                    "reduce gyro noise and improve flight performance"
                ),
                wiki_url=f"{_WIKI_BASE}/common-imu-notch-filtering.html",
            )
        ]
    return []


def check_failsafes(params: dict[str, float]) -> list[dict[str, Any]]:
    """Warn when critical failsafe actions are disabled."""
    warnings: list[dict[str, Any]] = []

    if params.get("BATT_FS_LOW_ACT", 1) == 0:
        warnings.append(
            _warning(
                severity="high",
                param="BATT_FS_LOW_ACT",
                value=0,
                issue="Battery low failsafe action is disabled",
                recommendation=(
                    "Set BATT_FS_LOW_ACT to land or RTL so the vehicle "
                    "returns home when the battery is low"
                ),
                wiki_url=f"{_WIKI_BASE}/failsafe-battery.html",
            )
        )

    if params.get("FS_THR_ENABLE", 1) == 0:
        warnings.append(
            _warning(
                severity="high",
                param="FS_THR_ENABLE",
                value=0,
                issue="Throttle failsafe is disabled",
                recommendation=(
                    "Enable the throttle failsafe to protect against "
                    "radio link loss"
                ),
                wiki_url=f"{_WIKI_BASE}/radio-failsafe.html",
            )
        )

    if params.get("FS_EKF_ACTION", 1) == 0:
        warnings.append(
            _warning(
                severity="medium",
                param="FS_EKF_ACTION",
                value=0,
                issue="EKF failsafe action is disabled",
                recommendation=(
                    "Set FS_EKF_ACTION to land or AltHold so the vehicle "
                    "reacts to EKF estimation failures"
                ),
                wiki_url=f"{_WIKI_BASE}/ekf-inav-failsafe.html",
            )
        )

    return warnings


def check_frame_config(params: dict[str, float]) -> list[dict[str, Any]]:
    if params.get("FRAME_CLASS", 1) == 0:
        return [
            _warning(
                severity="high",
                param="FRAME_CLASS",
                value=0,
                issue="Frame class is undefined",
                recommendation=(
                    "Set FRAME_CLASS to match your airframe (e.g. 1 for "
                    "Quad, 2 for Hexa) before flying"
                ),
                wiki_url=f"{_WIKI_BASE}/frame-type-configuration.html",
            )
        ]
    return []


def check_compass(params: dict[str, float]) -> list[dict[str, Any]]:
    if params.get("COMPASS_USE", 1) == 0:
        return [
            _warning(
                severity="high",
                param="COMPASS_USE",
                value=0,
                issue="Primary compass is disabled",
                recommendation=(
                    "Enable COMPASS_USE and perform a compass calibration "
                    "for reliable heading estimation"
                ),
                wiki_url=f"{_WIKI_BASE}/common-compass-calibration-in-mission-planner.html",
            )
        ]
    return []


def check_gps(params: dict[str, float]) -> list[dict[str, Any]]:
    if params.get("GPS_TYPE", 1) == 0:
        return [
            _warning(
                severity="high",
                param="GPS_TYPE",
                value=0,
                issue="GPS is disabled",
                recommendation=(
                    "Set GPS_TYPE to match your GPS receiver so the EKF "
                    "can use GPS position and velocity"
                ),
                wiki_url=f"{_WIKI_BASE}/common-installing-3dr-ublox-gps-compass-module.html",
            )
        ]
    return []


def check_ekf(params: dict[str, float]) -> list[dict[str, Any]]:
    # good enough for now — only checks threshold, not source config
    thresh = params.get("FS_EKF_THRESH", 0.8)
    if thresh > 1.0:
        return [
            _warning(
                severity="medium",
                param="FS_EKF_THRESH",
                value=thresh,
                issue="EKF failsafe threshold is very high — may never trigger",
                recommendation=(
                    "Lower FS_EKF_THRESH to 0.8 (default) so EKF failsafe "
                    "triggers before the estimate becomes dangerously bad"
                ),
                wiki_url=f"{_WIKI_BASE}/ekf-inav-failsafe.html",
            )
        ]
    return []


def check_arming(params: dict[str, float]) -> list[dict[str, Any]]:
    if params.get("ARMING_CHECK", 1) == 0:
        return [
            _warning(
                severity="high",
                param="ARMING_CHECK",
                value=0,
                issue="All pre-arm checks are disabled",
                recommendation=(
                    "Set ARMING_CHECK to 1 (all) to enable pre-arm safety "
                    "checks before every flight"
                ),
                wiki_url=f"{_WIKI_BASE}/prearm_safety_checks.html",
            )
        ]
    return []

# TODO: add checks for FENCE_ENABLE, GEO_FENCE stuff

_ALL_CHECKS = [
    check_tuning,
    check_notch_filter,
    check_failsafes,
    check_frame_config,
    check_compass,
    check_gps,
    check_ekf,
    check_arming,
]


def audit(params: dict[str, float]) -> list[dict[str, Any]]:
    """Run every parameter check and return a flat list of warnings."""
    warnings: list[dict[str, Any]] = []
    for check_fn in _ALL_CHECKS:
        warnings.extend(check_fn(params))
    return warnings
