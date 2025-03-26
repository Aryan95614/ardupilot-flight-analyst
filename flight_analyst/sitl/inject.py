"""SITL fault injection via MAVLink."""

import glob
import os
import time

from pymavlink import mavutil


class SITLInjector:
    """Connect to ArduPilot SITL and inject faults via parameter changes."""

    # TODO: add support for Plane/Rover SITL

    MODE_MAP = {
        "STABILIZE": 0,
        "ACRO": 1,
        "ALT_HOLD": 2,
        "AUTO": 3,
        "GUIDED": 4,
        "LOITER": 5,
        "RTL": 6,
        "CIRCLE": 7,
        "LAND": 9,
        "DRIFT": 11,
        "SPORT": 13,
        "FLIP": 14,
        "AUTOTUNE": 15,
        "POSHOLD": 16,
        "BRAKE": 17,
        "THROW": 18,
        "AVOID_ADSB": 19,
        "GUIDED_NOGPS": 20,
        "SMART_RTL": 21,
        "FLOWHOLD": 22,
        "FOLLOW": 23,
        "ZIGZAG": 24,
        "SYSTEMID": 25,
        "AUTOROTATE": 26,
    }

    FAULT_DEFAULTS = {
        "SIM_ENGINE_FAIL": 0,
        "SIM_ENGINE_MUL": 1.0,
        "SIM_GPS1_ENABLE": 1,
        "SIM_GPS1_GLTCH_X": 0,
        "SIM_GPS1_GLTCH_Y": 0,
        "SIM_MAG1_OFS_X": 5,
        "SIM_MAG1_OFS_Y": 13,
        "SIM_MAG1_OFS_Z": -18,
        "SIM_MAG1_FAIL": 0,
        "SIM_BATT_VOLTAGE": 12.6,
        "SIM_VIB_FREQ_X": 0,
        "SIM_VIB_FREQ_Y": 0,
        "SIM_VIB_FREQ_Z": 0,
        "SIM_ACC1_RND": 0,
        "SIM_RC_FAIL": 0,
        "SIM_ACC1_BIAS_X": 0,
        "SIM_ACC1_BIAS_Y": 0,
        "SIM_ACC1_BIAS_Z": 0,
        "SIM_WIND_SPD": 0,
        "SIM_WIND_TURB": 0,
    }

    def __init__(self, connection_string: str = "tcp:127.0.0.1:5760") -> None:
        self.conn_str = connection_string
        self.mav: mavutil.mavfile | None = None

    def connect(self) -> None:
        """Open a MAVLink connection and wait for a heartbeat."""
        self.mav = mavutil.mavlink_connection(self.conn_str, source_system=250)
        self.mav.wait_heartbeat()

    def set_param(self, name: str, value: float, timeout: float = 10) -> None:
        """Send PARAM_SET and block until PARAM_VALUE ack."""
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        self.mav.mav.param_set_send(
            self.mav.target_system,
            self.mav.target_component,
            name.encode("utf-8"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is None:
                continue
            received_name = msg.param_id
            if isinstance(received_name, bytes):
                received_name = received_name.decode("utf-8")
            received_name = received_name.rstrip("\x00")
            if received_name == name:
                return
        raise TimeoutError(f"Timed out waiting for PARAM_VALUE ack for {name}")

    def get_param(self, name: str, timeout: float = 10) -> float:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        self.mav.mav.param_request_read_send(
            self.mav.target_system,
            self.mav.target_component,
            name.encode("utf-8"),
            -1,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is None:
                continue
            received_name = msg.param_id
            if isinstance(received_name, bytes):
                received_name = received_name.decode("utf-8")
            received_name = received_name.rstrip("\x00")
            if received_name == name:
                return msg.param_value
        raise TimeoutError(f"Timed out waiting for PARAM_VALUE for {name}")

    def change_mode(self, mode_name: str) -> None:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        mode_id = self.MODE_MAP.get(mode_name.upper())
        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode_name}")

        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
            0, 0, 0, 0, 0,
        )
        self.wait_mode(mode_name)

    def arm(self, timeout: float = 30) -> None:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                return
        raise TimeoutError("Timed out waiting for armed state")

    def disarm(self, force: bool = False, timeout: float = 30) -> None:
        """Disarm the vehicle.  *force* sends the magic p2 override."""
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        p2 = 21196 if force else 0
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, p2, 0, 0, 0, 0, 0,
        )
        self.wait_disarmed(timeout=timeout)

    def takeoff(self, target_alt: float = 30, timeout: float = 60) -> None:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0, target_alt,
        )
        self.wait_alt(target_alt * 0.9, target_alt * 1.2, timeout=timeout)

    def wait_alt(self, alt_min: float, alt_max: float, timeout: float = 60) -> None:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=1,
            )
            if msg is None:
                continue
            alt_m = msg.relative_alt / 1000.0  # mm -> m
            if alt_min <= alt_m <= alt_max:
                return
        raise TimeoutError(
            f"Timed out waiting for altitude in [{alt_min}, {alt_max}]"
        )

    def wait_mode(self, mode_name: str, timeout: float = 60) -> None:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        target_id = self.MODE_MAP.get(mode_name.upper())
        if target_id is None:
            raise ValueError(f"Unknown mode: {mode_name}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.custom_mode == target_id:
                return
        raise TimeoutError(f"Timed out waiting for mode {mode_name}")

    def wait_disarmed(self, timeout: float = 120) -> None:
        if self.mav is None:
            raise RuntimeError("Not connected – call connect() first")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg is None:
                continue
            if not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                return
        raise TimeoutError("Timed out waiting for disarmed state")

    # -- fault injection --

    def inject_motor_failure(self, motor: int, efficiency: float = 0.0) -> None:
        self.set_param("SIM_ENGINE_FAIL", 1 << (motor - 1))
        self.set_param("SIM_ENGINE_MUL", efficiency)

    def inject_gps_glitch(
        self,
        lat_offset: float = 0.001,
        lon_offset: float = 0.001,
    ) -> None:
        self.set_param("SIM_GPS1_GLTCH_X", lat_offset)
        self.set_param("SIM_GPS1_GLTCH_Y", lon_offset)

    def inject_gps_loss(self) -> None:
        self.set_param("SIM_GPS1_ENABLE", 0)

    def inject_compass_interference(self, offset: float = 200) -> None:
        self.set_param("SIM_MAG1_OFS_X", offset)
        self.set_param("SIM_MAG1_OFS_Y", offset)
        self.set_param("SIM_MAG1_OFS_Z", offset)

    def inject_compass_failure(self) -> None:
        self.set_param("SIM_MAG1_FAIL", 1)

    def inject_battery_drop(self, voltage: float) -> None:
        self.set_param("SIM_BATT_VOLTAGE", voltage)

    def inject_vibration(self, freq: float = 150, noise: float = 10) -> None:
        self.set_param("SIM_VIB_FREQ_X", freq)
        self.set_param("SIM_VIB_FREQ_Y", freq)
        self.set_param("SIM_VIB_FREQ_Z", freq)
        self.set_param("SIM_ACC1_RND", noise)

    def inject_rc_failsafe(self) -> None:
        self.set_param("SIM_RC_FAIL", 1)

    def inject_imu_bias(self, accel_bias: float = 5.0) -> None:
        self.set_param("SIM_ACC1_BIAS_X", accel_bias)
        self.set_param("SIM_ACC1_BIAS_Y", accel_bias)
        self.set_param("SIM_ACC1_BIAS_Z", accel_bias)

    def inject_wind(self, speed: float, turbulence: float = 0) -> None:
        self.set_param("SIM_WIND_SPD", speed)
        self.set_param("SIM_WIND_TURB", turbulence)

    def clear_all_faults(self) -> None:
        for name, value in self.FAULT_DEFAULTS.items():
            self.set_param(name, value)

    @staticmethod
    def get_latest_log(log_dir: str = "logs") -> str | None:
        """Return path to the newest .BIN log, or None."""
        logs = sorted(glob.glob(os.path.join(log_dir, "*.BIN")))
        return logs[-1] if logs else None
