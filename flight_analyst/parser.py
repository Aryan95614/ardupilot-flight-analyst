"""LogParser — pymavlink wrapper for diagnostic use."""

import re
from pathlib import Path
from typing import Any

import numpy as np
from pymavlink import mavutil

_EKF3_HEALTH_MSG = "XKF4"
_EKF2_HEALTH_MSGS = ("NKF4", "EKF4")

_VEHICLE_KEYWORDS = {
    "ArduCopter": "Copter",
    "Copter": "Copter",
    "ArduPlane": "Plane",
    "Plane": "Plane",
    "APMrover2": "Rover",
    "Rover": "Rover",
    "ArduSub": "Sub",
    "Sub": "Sub",
    "AntennaTracker": "Tracker",
    "Tracker": "Tracker",
    "Blimp": "Blimp",
}

# FRAME_CLASS parameter values -> rough vehicle type (copter-centric)
_FRAME_CLASS_MAP = {
    0: "Undefined",
    1: "Copter",   # Quad
    2: "Copter",   # Hexa
    3: "Copter",   # Octa
    4: "Copter",   # OctaQuad
    5: "Copter",   # Y6
    6: "Copter",   # Heli
    7: "Copter",   # Tri
    13: "Copter",  # Heli Dual
}


class LogParser:
    """Parse an ArduPilot .bin or .log file for diagnostic analysis."""

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = str(filepath)
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._preload()

    def _preload(self) -> None:
        """Single-pass read: cache every message keyed by type name."""
        mlog = mavutil.mavlink_connection(self._filepath)
        while True:
            msg = mlog.recv_match(blocking=False)
            if msg is None:
                break
            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                continue
            fields = msg.to_dict()
            ts = getattr(msg, "_timestamp", None)
            if ts is not None:
                fields["_timestamp"] = float(ts)
            self._cache.setdefault(msg_type, []).append(fields)

    # --- public query API ---

    def get_messages(self, msg_type: str) -> list[dict[str, Any]]:
        """Return all cached messages of the given type (empty list if absent)."""
        return list(self._cache.get(msg_type, []))

    def get_time_series(
        self, msg_type: str, field: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) numpy arrays for a given field.

        Timestamps come from TimeUS (converted to seconds) if present,
        otherwise from _timestamp. Raises KeyError when msg_type has no
        messages, and ValueError when field is missing.
        """
        msgs = self._cache.get(msg_type)
        if not msgs:
            raise KeyError(f"No messages of type '{msg_type}' in log")

        if field not in msgs[0]:
            raise ValueError(
                f"Field '{field}' not found in '{msg_type}' messages"
            )

        times: list[float] = []
        values: list[float] = []
        for m in msgs:
            if "TimeUS" in m:
                times.append(float(m["TimeUS"]) * 1e-6)
            elif "_timestamp" in m:
                times.append(float(m["_timestamp"]))
            else:
                times.append(0.0)
            values.append(float(m[field]))

        return np.array(times), np.array(values)

    def get_params(self) -> dict[str, float]:
        """Return {name: value} for every PARM message."""
        return {
            m["Name"]: float(m["Value"])
            for m in self._cache.get("PARM", [])
        }

    def get_flight_modes(self) -> list[dict[str, Any]]:
        return list(self._cache.get("MODE", []))

    def get_errors(self) -> list[dict[str, Any]]:
        return list(self._cache.get("ERR", []))

    def get_events(self) -> list[dict[str, Any]]:
        return list(self._cache.get("EV", []))

    def get_firmware_version(self) -> str | None:
        """Best-effort firmware version string.

        Checks VER messages first, then falls back to scanning MSG
        messages for a version-like pattern.
        """
        for ver in self._cache.get("VER", []):
            for key in ("APJ", "FWVer"):
                if key in ver:
                    return str(ver[key])

        for msg in self._cache.get("MSG", []):
            text = msg.get("Message", "")
            match = re.search(r"[Vv]?\d+\.\d+\.\d+", text)
            if match:
                return match.group(0)

        return None

    def get_ekf_version(self) -> int | None:
        """Return 2 or 3 depending on which EKF health message is present."""
        if _EKF3_HEALTH_MSG in self._cache:
            return 3
        for t in _EKF2_HEALTH_MSGS:
            if t in self._cache:
                return 2
        return None

    def get_ekf_health_msg_type(self) -> str | None:
        """Return the actual message type used for EKF health, or None."""
        if _EKF3_HEALTH_MSG in self._cache:
            return _EKF3_HEALTH_MSG
        for t in _EKF2_HEALTH_MSGS:
            if t in self._cache:
                return t
        return None

    def get_vehicle_type(self) -> str | None:
        """Auto-detect vehicle type from MSG boot strings or FRAME_CLASS."""
        for msg in self._cache.get("MSG", []):
            text = msg.get("Message", "")
            for keyword, vtype in _VEHICLE_KEYWORDS.items():
                if keyword in text:
                    return vtype

        # Fallback: FRAME_CLASS parameter
        params = self.get_params()
        fc = params.get("FRAME_CLASS")
        if fc is not None:
            return _FRAME_CLASS_MAP.get(int(fc), "Unknown")

        return None

    def get_flight_duration(self) -> float:
        """Estimated flight duration in seconds from first to last timestamp."""
        first_ts: float | None = None
        last_ts: float | None = None
        for msgs in self._cache.values():
            for m in msgs:
                ts = m.get("_timestamp")
                if ts is None:
                    continue
                ts = float(ts)
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
        if first_ts is not None and last_ts is not None:
            return last_ts - first_ts
        return 0.0

    def get_message_types(self) -> list[str]:
        """Return sorted list of all message types present in the log."""
        return sorted(self._cache.keys())

    def has_message_type(self, msg_type: str) -> bool:
        return msg_type in self._cache and len(self._cache[msg_type]) > 0
