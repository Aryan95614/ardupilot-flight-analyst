"""Feature extraction for ArduPilot flight logs."""

import math
from typing import Any
import numpy as np
from flight_analyst.parser import LogParser

# lookup tables

ERR_SUBSYS = {
    1: "Main",
    2: "Radio",
    3: "Compass",
    5: "Radio/FS",
    6: "EKF/InertialNav",
    7: "BaroGlitch",
    10: "FlightMode",
    12: "CrashCheck",
    15: "Parachute",
    16: "EKF/InertialNav",
    17: "EKF_primary",
    18: "Terrain",
    22: "GPS_Glitch",
    26: "EKF_primary",
    27: "EKF_lane_switch",
}

EKF_FAULT_BITS = {
    0: "bad_mag_x",
    1: "bad_mag_y",
    2: "bad_mag_z",
    3: "bad_airspeed",
    4: "bad_sideslip",
    5: "bad_optflow_x",
    6: "bad_optflow_y",
}

EKF_TIMEOUT_BITS = {
    0: "velocity",
    1: "position_horiz",
    2: "position_vert",
    3: "height",
    4: "magnetometer",
    5: "airspeed",
}


def _safe_stat(values: np.ndarray, func_name: str) -> float:
    if len(values) == 0:
        return float("nan")
    return float(getattr(np, func_name)(values))


def _decode_bitmask(value: int, bit_table: dict[int, str]) -> list[str]:
    flags: list[str] = []
    for bit, name in bit_table.items():
        if value & (1 << bit):
            flags.append(name)
    return flags


# ---

class FeatureExtractor:
    """Extract flight-analysis features from a parsed ArduPilot log."""

    def __init__(self, parser: LogParser) -> None:
        self.parser = parser
        self._ekf_msg: str | None = None

    @staticmethod
    def _time_above_threshold(
        ts: np.ndarray, values: np.ndarray, threshold: float
    ) -> float:
        """Total seconds that values exceed threshold (trapezoidal estimation)."""
        if len(ts) < 2:
            return 0.0
        above = values > threshold
        both = above[:-1] & above[1:]
        dt = np.diff(ts)
        return float(np.sum(dt[both]))

    @staticmethod
    def _first_above_threshold(
        ts: np.ndarray, values: np.ndarray, threshold: float
    ) -> float:
        """Timestamp of first sample exceeding threshold, or NaN."""
        indices = np.where(values > threshold)[0]
        if len(indices) == 0:
            return float("nan")
        return float(ts[indices[0]])

    def _get_msgs(self, msg_type: str) -> list[Any]:
        return self.parser.get_messages(msg_type)

    def _detect_ekf_msg(self) -> str:
        if self._ekf_msg is not None:
            return self._ekf_msg
        self._ekf_msg = self.parser.get_ekf_health_msg_type()
        return self._ekf_msg

    # --- extractors ---

    def vibration_features(self) -> dict[str, Any]:
        """VIBE message features: vibration levels and clipping events."""
        msgs = self._get_msgs("VIBE")
        if not msgs:
            return {"vibe_available": False}

        ts = np.array([m.TimeUS * 1e-6 for m in msgs])
        vx = np.array([m.VibeX for m in msgs])
        vy = np.array([m.VibeY for m in msgs])
        vz = np.array([m.VibeZ for m in msgs])
        clips = np.array(
            [getattr(m, "Clip0", 0) + getattr(m, "Clip1", 0) + getattr(m, "Clip2", 0) for m in msgs]
        )

        magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

        if len(ts) >= 2:
            d_mag = np.diff(magnitude)
            d_t = np.diff(ts)
            safe_dt = np.where(d_t == 0, 1e-9, d_t)
            roc = d_mag / safe_dt
            max_roc = float(np.max(np.abs(roc)))
        else:
            max_roc = 0.0

        return {
            "vibe_available": True,
            "vibe_x_max": float(np.max(vx)),
            "vibe_x_mean": float(np.mean(vx)),
            "vibe_x_std": float(np.std(vx)),
            "vibe_y_max": float(np.max(vy)),
            "vibe_y_mean": float(np.mean(vy)),
            "vibe_y_std": float(np.std(vy)),
            "vibe_z_max": float(np.max(vz)),
            "vibe_z_mean": float(np.mean(vz)),
            "vibe_z_std": float(np.std(vz)),
            "vibe_clip_total": int(np.sum(clips)),
            "vibe_time_above_30": self._time_above_threshold(ts, magnitude, 30.0),
            "vibe_magnitude_max_roc": max_roc,
        }

    def ekf_features(self) -> dict[str, Any]:
        """EKF health features: innovation variances, faults, timeouts."""
        ekf_type = self._detect_ekf_msg()
        msgs = self._get_msgs(ekf_type)
        if not msgs:
            return {"ekf_available": False}

        ts = np.array([m.TimeUS * 1e-6 for m in msgs])
        sv = np.array([m.SV for m in msgs])
        sp = np.array([m.SP for m in msgs])
        sh = np.array([m.SH for m in msgs])
        sm = np.array([m.SM for m in msgs])

        # ArduPilot default thresholds
        sv_thresh = 0.8
        sp_thresh = 0.8
        sh_thresh = 0.8
        sm_thresh = 0.8

        fs_values = [getattr(m, "FS", 0) for m in msgs]
        ts_values = [getattr(m, "TS", 0) for m in msgs]

        # OR all bitmask values across flight
        fs_union = 0
        for v in fs_values:
            fs_union |= int(v)
        ts_union = 0
        for v in ts_values:
            ts_union |= int(v)

        fault_flags = _decode_bitmask(fs_union, EKF_FAULT_BITS)
        timeout_flags = _decode_bitmask(ts_union, EKF_TIMEOUT_BITS)

        return {
            "ekf_available": True,
            "ekf_msg_type": ekf_type,
            "ekf_sv_max": float(np.max(sv)),
            "ekf_sv_mean": float(np.mean(sv)),
            "ekf_sp_max": float(np.max(sp)),
            "ekf_sp_mean": float(np.mean(sp)),
            "ekf_sh_max": float(np.max(sh)),
            "ekf_sh_mean": float(np.mean(sh)),
            "ekf_sm_max": float(np.max(sm)),
            "ekf_sm_mean": float(np.mean(sm)),
            "ekf_sv_exceedance_count": int(np.sum(sv > sv_thresh)),
            "ekf_sp_exceedance_count": int(np.sum(sp > sp_thresh)),
            "ekf_sh_exceedance_count": int(np.sum(sh > sh_thresh)),
            "ekf_sm_exceedance_count": int(np.sum(sm > sm_thresh)),
            "ekf_sv_first_exceedance_time": self._first_above_threshold(ts, sv, sv_thresh),
            "ekf_sp_first_exceedance_time": self._first_above_threshold(ts, sp, sp_thresh),
            "ekf_fault_flags": fault_flags,
            "ekf_timeout_flags": timeout_flags,
            "ekf_fs_raw": fs_union,
            "ekf_ts_raw": ts_union,
        }

    def attitude_features(self) -> dict[str, Any]:
        """ATT message features: attitude errors and sustained anomalies."""
        msgs = self._get_msgs("ATT")
        if not msgs:
            return {"att_available": False}

        ts = np.array([m.TimeUS * 1e-6 for m in msgs])
        err_rp = np.array([m.ErrRP for m in msgs])
        err_yaw = np.array([m.ErrYaw for m in msgs])
        roll = np.array([m.Roll for m in msgs])
        pitch = np.array([m.Pitch for m in msgs])
        des_roll = np.array([getattr(m, "DesRoll", 0.0) for m in msgs])
        des_pitch = np.array([getattr(m, "DesPitch", 0.0) for m in msgs])

        roll_error = np.abs(roll - des_roll)
        pitch_error = np.abs(pitch - des_pitch)

        err_rp_thresh = 0.5

        return {
            "att_available": True,
            "att_err_rp_max": float(np.max(err_rp)),
            "att_err_rp_mean": float(np.mean(err_rp)),
            "att_err_yaw_max": float(np.max(err_yaw)),
            "att_err_yaw_mean": float(np.mean(err_yaw)),
            "att_roll_error_max": float(np.max(roll_error)),
            "att_roll_error_mean": float(np.mean(roll_error)),
            "att_pitch_error_max": float(np.max(pitch_error)),
            "att_pitch_error_mean": float(np.mean(pitch_error)),
            "att_sustained_err_rp_time": self._time_above_threshold(ts, err_rp, err_rp_thresh),
            "att_first_err_rp_exceedance": self._first_above_threshold(ts, err_rp, err_rp_thresh),
        }

    def gps_features(self) -> dict[str, Any]:
        """GPS message features: sat count, HDOP, fix losses."""
        msgs = self._get_msgs("GPS")
        if not msgs:
            return {"gps_available": False}

        hdop = np.array([m.HDop for m in msgs])
        nsats = np.array([m.NSats for m in msgs])
        status = np.array([m.Status for m in msgs])

        fix_loss_count = 0
        for i in range(1, len(status)):
            if status[i - 1] >= 3 and status[i] < 3:
                fix_loss_count += 1

        return {
            "gps_available": True,
            "gps_hdop_max": float(np.max(hdop)),
            "gps_hdop_mean": float(np.mean(hdop)),
            "gps_nsats_min": int(np.min(nsats)),
            "gps_nsats_mean": float(np.mean(nsats)),
            "gps_fix_loss_count": fix_loss_count,
        }

    def battery_features(self) -> dict[str, Any]:
        """BAT features: voltage curve, current draw."""
        msgs = self._get_msgs("BAT")
        if not msgs:
            return {"bat_available": False}

        ts = np.array([m.TimeUS * 1e-6 for m in msgs])
        volt = np.array([m.Volt for m in msgs])
        curr = np.array([m.Curr for m in msgs])

        if len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
            volt_drop_rate = float((volt[0] - volt[-1]) / (ts[-1] - ts[0]))
        else:
            volt_drop_rate = 0.0

        return {
            "bat_available": True,
            "bat_volt_min": float(np.min(volt)),
            "bat_volt_max": float(np.max(volt)),
            "bat_volt_mean": float(np.mean(volt)),
            "bat_volt_drop_rate": volt_drop_rate,
            "bat_curr_max": float(np.max(curr)),
            "bat_curr_mean": float(np.mean(curr)),
        }

    def motor_features(self) -> dict[str, Any]:
        """RCOU features: per-channel stats and cross-channel spread."""
        msgs = self._get_msgs("RCOU")
        if not msgs:
            return {"motor_available": False}

        channel_names: list[str] = []
        sample = msgs[0]
        for i in range(1, 17):
            attr = f"C{i}"
            if hasattr(sample, attr):
                channel_names.append(attr)

        if not channel_names:
            return {"motor_available": False}

        channels: dict[str, np.ndarray] = {}
        for ch in channel_names:
            channels[ch] = np.array([getattr(m, ch) for m in msgs])

        result: dict[str, Any] = {"motor_available": True}

        for ch, vals in channels.items():
            result[f"motor_{ch}_mean"] = float(np.mean(vals))
            result[f"motor_{ch}_max"] = float(np.max(vals))
            result[f"motor_{ch}_std"] = float(np.std(vals))

        ch_means = np.array([float(np.mean(v)) for v in channels.values()])
        result["motor_cross_channel_spread"] = float(np.std(ch_means))

        # saturation = any channel near max PWM (2000)
        saturation_threshold = 1950
        any_saturated = np.zeros(len(msgs), dtype=bool)
        for vals in channels.values():
            any_saturated |= vals >= saturation_threshold
        result["motor_saturation_count"] = int(np.sum(any_saturated))

        return result

    def power_features(self) -> dict[str, Any]:
        """POWR features: board voltage and brownout risk."""
        msgs = self._get_msgs("POWR")
        if not msgs:
            return {"powr_available": False}

        vcc = np.array([m.Vcc for m in msgs])

        brownout_threshold = 4.5
        brownout_count = int(np.sum(vcc < brownout_threshold))

        return {
            "powr_available": True,
            "powr_vcc_min": float(np.min(vcc)),
            "powr_vcc_mean": float(np.mean(vcc)),
            "powr_vcc_std": float(np.std(vcc)),
            "powr_brownout_risk_count": brownout_count,
        }

    def performance_features(self) -> dict[str, Any]:
        """PM features: scheduling, loop time, CPU load."""
        msgs = self._get_msgs("PM")
        if not msgs:
            return {"pm_available": False}

        nlon = np.array([m.NLon for m in msgs])
        max_t = np.array([m.MaxT for m in msgs])
        load = np.array([getattr(m, "Load", 0) for m in msgs])

        overrun_count = int(np.sum(nlon > 0))

        return {
            "pm_available": True,
            "pm_nlon_max": int(np.max(nlon)),
            "pm_nlon_total": int(np.sum(nlon)),
            "pm_max_t_max": float(np.max(max_t)),
            "pm_max_t_mean": float(np.mean(max_t)),
            "pm_load_max": float(np.max(load)),
            "pm_load_mean": float(np.mean(load)),
            "pm_overrun_count": overrun_count,
        }

    def control_features(self) -> dict[str, Any]:
        """CTUN features: throttle output and altitude tracking."""
        msgs = self._get_msgs("CTUN")
        if not msgs:
            return {"ctun_available": False}

        tho = np.array([m.ThO for m in msgs])
        alt = np.array([getattr(m, "Alt", 0.0) for m in msgs])
        des_alt = np.array([getattr(m, "DAlt", 0.0) for m in msgs])

        alt_error = np.abs(alt - des_alt)

        return {
            "ctun_available": True,
            "ctun_tho_max": float(np.max(tho)),
            "ctun_tho_mean": float(np.mean(tho)),
            "ctun_alt_error_max": float(np.max(alt_error)),
            "ctun_alt_error_mean": float(np.mean(alt_error)),
        }

    def rc_features(self) -> dict[str, Any]:
        """RCIN features: pilot activity and failsafe detection."""
        msgs = self._get_msgs("RCIN")
        if not msgs:
            return {"rcin_available": False}

        ch3 = np.array([getattr(m, "C3", 1500) for m in msgs])

        # std-dev of throttle as a proxy for pilot activity
        pilot_activity = float(np.std(ch3))

        # TODO: maybe also track Ch5 for flight mode switches
        fs_threshold = 900
        fs_count = int(np.sum(ch3 < fs_threshold))

        return {
            "rcin_available": True,
            "rcin_pilot_activity": pilot_activity,
            "rcin_throttle_mean": float(np.mean(ch3)),
            "rcin_failsafe_count": fs_count,
        }

    def error_features(self) -> dict[str, Any]:
        """ERR/EV/MODE message features."""
        err_msgs = self._get_msgs("ERR")
        ev_msgs = self._get_msgs("EV")
        mode_msgs = self._get_msgs("MODE")

        if not err_msgs and not ev_msgs and not mode_msgs:
            return {"err_available": False}

        result: dict[str, Any] = {"err_available": True}

        decoded_errors: list[dict[str, Any]] = []
        crash_check_count = 0
        ekf_failsafe_count = 0

        for m in err_msgs:
            subsys = int(m.Subsys)
            ecode = int(m.ECode)
            subsys_name = ERR_SUBSYS.get(subsys, f"Unknown({subsys})")
            decoded_errors.append(
                {"subsys": subsys, "subsys_name": subsys_name, "ecode": ecode}
            )
            if subsys == 12:
                crash_check_count += 1
            if subsys in (6, 16, 17, 26):
                ekf_failsafe_count += 1

        result["err_decoded"] = decoded_errors
        result["err_total_count"] = len(err_msgs)
        result["err_crash_check_count"] = crash_check_count
        result["err_ekf_failsafe_count"] = ekf_failsafe_count

        result["ev_count"] = len(ev_msgs)
        if ev_msgs:
            result["ev_ids"] = [int(m.Id) for m in ev_msgs]
        else:
            result["ev_ids"] = []

        result["mode_change_count"] = len(mode_msgs)
        if mode_msgs:
            result["modes"] = [getattr(m, "Mode", None) for m in mode_msgs]
        else:
            result["modes"] = []

        return result

    def mag_features(self) -> dict[str, Any]:
        """MAG features: field magnitude and interference."""
        msgs = self._get_msgs("MAG")
        if not msgs:
            return {"mag_available": False}

        mx = np.array([m.MagX for m in msgs])
        my = np.array([m.MagY for m in msgs])
        mz = np.array([m.MagZ for m in msgs])

        magnitude = np.sqrt(mx**2 + my**2 + mz**2)

        ofsx = np.array([getattr(m, "OfsX", 0.0) for m in msgs])
        ofsy = np.array([getattr(m, "OfsY", 0.0) for m in msgs])
        ofsz = np.array([getattr(m, "OfsZ", 0.0) for m in msgs])
        ofs_magnitude = np.sqrt(ofsx**2 + ofsy**2 + ofsz**2)

        return {
            "mag_available": True,
            "mag_field_max": float(np.max(magnitude)),
            "mag_field_min": float(np.min(magnitude)),
            "mag_field_mean": float(np.mean(magnitude)),
            "mag_field_range": float(np.max(magnitude) - np.min(magnitude)),
            "mag_interference_max": float(np.max(ofs_magnitude)),
            "mag_interference_mean": float(np.mean(ofs_magnitude)),
        }

    _EXTRACTORS: list[str] = [
        "vibration_features",
        "ekf_features",
        "attitude_features",
        "gps_features",
        "battery_features",
        "motor_features",
        "power_features",
        "performance_features",
        "control_features",
        "rc_features",
        "error_features",
        "mag_features",
    ]

    def extract_all(self) -> dict[str, Any]:
        """Run every extractor and merge results into one dict.

        Individual failures are stored under _error_<name> keys.
        """
        combined: dict[str, Any] = {}
        for name in self._EXTRACTORS:
            try:
                result = getattr(self, name)()
                combined.update(result)
            except Exception as exc:
                combined[f"_error_{name}"] = f"{type(exc).__name__}: {exc}"
        return combined
