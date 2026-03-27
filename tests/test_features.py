"""FeatureExtractor tests."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flight_analyst.features import (
    EKF_FAULT_BITS,
    EKF_TIMEOUT_BITS,
    ERR_SUBSYS,
    FeatureExtractor,
    _decode_bitmask,
)


def _make_msg(**kwargs) -> dict:
    """Create a mock message dict matching what LogParser returns."""
    return dict(**kwargs)


def _make_parser(messages: dict[str, list] | None = None, ekf_type: str = "NKF4"):
    parser = MagicMock()
    messages = messages or {}

    def get_messages(msg_type: str):
        return messages.get(msg_type, [])

    parser.get_messages = MagicMock(side_effect=get_messages)
    parser.get_ekf_health_msg_type = MagicMock(return_value=ekf_type)
    return parser


def _make_extractor(messages=None, ekf_type="NKF4"):
    parser = _make_parser(messages, ekf_type)
    return FeatureExtractor(parser)


class TestVibrationFeatures:
    def test_no_vibe_messages(self):
        fe = _make_extractor()
        result = fe.vibration_features()
        assert result["vibe_available"] is False

    def test_basic_vibe(self):
        msgs = [
            _make_msg(TimeUS=1_000_000, VibeX=10, VibeY=20, VibeZ=15, Clip0=1, Clip1=0, Clip2=0),
            _make_msg(TimeUS=2_000_000, VibeX=12, VibeY=22, VibeZ=18, Clip0=0, Clip1=1, Clip2=0),
            _make_msg(TimeUS=3_000_000, VibeX=11, VibeY=21, VibeZ=16, Clip0=0, Clip1=0, Clip2=2),
        ]
        fe = _make_extractor({"VIBE": msgs})
        result = fe.vibration_features()

        assert result["vibe_available"] is True
        assert result["vibe_x_max"] == 12
        assert result["vibe_y_max"] == 22
        assert result["vibe_z_max"] == 18
        assert result["vibe_clip_total"] == 4  # 1 + 1 + 2
        assert isinstance(result["vibe_x_std"], float)

    def test_time_above_30(self):
        # magnitude = sqrt(30^2 + 30^2 + 30^2) ~ 51.96 > 30
        msgs = [
            _make_msg(TimeUS=0, VibeX=30, VibeY=30, VibeZ=30, Clip0=0, Clip1=0, Clip2=0),
            _make_msg(TimeUS=1_000_000, VibeX=30, VibeY=30, VibeZ=30, Clip0=0, Clip1=0, Clip2=0),
            _make_msg(TimeUS=2_000_000, VibeX=1, VibeY=1, VibeZ=1, Clip0=0, Clip1=0, Clip2=0),
        ]
        fe = _make_extractor({"VIBE": msgs})
        result = fe.vibration_features()
        assert result["vibe_time_above_30"] == pytest.approx(1.0, abs=0.01)

    def test_rate_of_change(self):
        msgs = [
            _make_msg(TimeUS=0, VibeX=0, VibeY=0, VibeZ=0, Clip0=0, Clip1=0, Clip2=0),
            _make_msg(TimeUS=1_000_000, VibeX=100, VibeY=0, VibeZ=0, Clip0=0, Clip1=0, Clip2=0),
        ]
        fe = _make_extractor({"VIBE": msgs})
        result = fe.vibration_features()
        # magnitude 0 -> 100 in 1s
        assert result["vibe_magnitude_max_roc"] == pytest.approx(100.0, rel=0.01)


class TestEKFFeatures:
    def test_no_ekf_messages(self):
        fe = _make_extractor(ekf_type="XKF4")
        result = fe.ekf_features()
        assert result["ekf_available"] is False

    def test_basic_ekf_nkf4(self):
        msgs = [
            _make_msg(TimeUS=1_000_000, SV=0.5, SP=0.3, SH=0.2, SM=0.1, FS=0, TS=0),
            _make_msg(TimeUS=2_000_000, SV=0.9, SP=0.4, SH=0.3, SM=0.2, FS=0, TS=0),
        ]
        fe = _make_extractor({"NKF4": msgs}, ekf_type="NKF4")
        result = fe.ekf_features()

        assert result["ekf_available"] is True
        assert result["ekf_msg_type"] == "NKF4"
        assert result["ekf_sv_max"] == pytest.approx(0.9)
        assert result["ekf_sv_exceedance_count"] == 1  # 0.9 > 0.8

    def test_ekf_xkf4(self):
        msgs = [
            _make_msg(TimeUS=0, SV=0.1, SP=0.1, SH=0.1, SM=0.1, FS=0, TS=0),
        ]
        fe = _make_extractor({"XKF4": msgs}, ekf_type="XKF4")
        result = fe.ekf_features()
        assert result["ekf_msg_type"] == "XKF4"
        assert result["ekf_sv_exceedance_count"] == 0

    def test_fault_bitmask_decoding(self):
        # FS = 0b00001001 -> bit0 (bad_mag_x) + bit3 (bad_airspeed)
        msgs = [
            _make_msg(TimeUS=0, SV=0.1, SP=0.1, SH=0.1, SM=0.1, FS=0b00001001, TS=0),
        ]
        fe = _make_extractor({"NKF4": msgs})
        result = fe.ekf_features()
        assert "bad_mag_x" in result["ekf_fault_flags"]
        assert "bad_airspeed" in result["ekf_fault_flags"]
        assert len(result["ekf_fault_flags"]) == 2

    def test_timeout_bitmask_decoding(self):
        # TS = 0b00000110 -> bit1 (position_horiz) + bit2 (position_vert)
        msgs = [
            _make_msg(TimeUS=0, SV=0.1, SP=0.1, SH=0.1, SM=0.1, FS=0, TS=0b00000110),
        ]
        fe = _make_extractor({"NKF4": msgs})
        result = fe.ekf_features()
        assert "position_horiz" in result["ekf_timeout_flags"]
        assert "position_vert" in result["ekf_timeout_flags"]
        assert result["ekf_ts_raw"] == 0b00000110

    def test_first_exceedance_time(self):
        msgs = [
            _make_msg(TimeUS=1_000_000, SV=0.5, SP=0.5, SH=0.5, SM=0.5, FS=0, TS=0),
            _make_msg(TimeUS=3_000_000, SV=0.9, SP=0.5, SH=0.5, SM=0.5, FS=0, TS=0),
            _make_msg(TimeUS=5_000_000, SV=0.95, SP=0.5, SH=0.5, SM=0.5, FS=0, TS=0),
        ]
        fe = _make_extractor({"NKF4": msgs})
        result = fe.ekf_features()
        assert result["ekf_sv_first_exceedance_time"] == pytest.approx(3.0)

    def test_fault_union_across_messages(self):
        msgs = [
            _make_msg(TimeUS=0, SV=0.1, SP=0.1, SH=0.1, SM=0.1, FS=0b001, TS=0b010),
            _make_msg(TimeUS=1_000_000, SV=0.1, SP=0.1, SH=0.1, SM=0.1, FS=0b100, TS=0b001),
        ]
        fe = _make_extractor({"NKF4": msgs})
        result = fe.ekf_features()
        assert result["ekf_fs_raw"] == 0b101
        assert result["ekf_ts_raw"] == 0b011


class TestAttitudeFeatures:
    def test_no_att_messages(self):
        fe = _make_extractor()
        result = fe.attitude_features()
        assert result["att_available"] is False

    def test_basic_attitude(self):
        msgs = [
            _make_msg(TimeUS=0, ErrRP=0.1, ErrYaw=0.05, Roll=2.0, Pitch=1.0, DesRoll=2.0, DesPitch=1.0),
            _make_msg(TimeUS=1_000_000, ErrRP=1.0, ErrYaw=0.5, Roll=10.0, Pitch=5.0, DesRoll=2.0, DesPitch=1.0),
        ]
        fe = _make_extractor({"ATT": msgs})
        result = fe.attitude_features()
        assert result["att_available"] is True
        assert result["att_err_rp_max"] == pytest.approx(1.0)
        assert result["att_roll_error_max"] == pytest.approx(8.0)
        assert result["att_pitch_error_max"] == pytest.approx(4.0)

    def test_sustained_error(self):
        msgs = [
            _make_msg(TimeUS=0, ErrRP=0.6, ErrYaw=0.0, Roll=0, Pitch=0, DesRoll=0, DesPitch=0),
            _make_msg(TimeUS=1_000_000, ErrRP=0.7, ErrYaw=0.0, Roll=0, Pitch=0, DesRoll=0, DesPitch=0),
            _make_msg(TimeUS=2_000_000, ErrRP=0.1, ErrYaw=0.0, Roll=0, Pitch=0, DesRoll=0, DesPitch=0),
        ]
        fe = _make_extractor({"ATT": msgs})
        result = fe.attitude_features()
        # First two above 0.5 threshold -> 1 second sustained
        assert result["att_sustained_err_rp_time"] == pytest.approx(1.0)


class TestGPSFeatures:
    def test_no_gps_messages(self):
        fe = _make_extractor()
        result = fe.gps_features()
        assert result["gps_available"] is False

    def test_basic_gps(self):
        msgs = [
            _make_msg(HDop=1.2, NSats=12, Status=3),
            _make_msg(HDop=1.5, NSats=10, Status=3),
            _make_msg(HDop=2.0, NSats=8, Status=2),
        ]
        fe = _make_extractor({"GPS": msgs})
        result = fe.gps_features()
        assert result["gps_available"] is True
        assert result["gps_hdop_max"] == pytest.approx(2.0)
        assert result["gps_nsats_min"] == 8
        assert result["gps_fix_loss_count"] == 1  # 3->2 transition


class TestBatteryFeatures:
    def test_no_bat_messages(self):
        fe = _make_extractor()
        result = fe.battery_features()
        assert result["bat_available"] is False

    def test_voltage_drop_rate(self):
        msgs = [
            _make_msg(TimeUS=0, Volt=12.6, Curr=5.0),
            _make_msg(TimeUS=10_000_000, Volt=12.0, Curr=6.0),
        ]
        fe = _make_extractor({"BAT": msgs})
        result = fe.battery_features()
        assert result["bat_available"] is True
        # 0.6V over 10s = 0.06 V/s
        assert result["bat_volt_drop_rate"] == pytest.approx(0.06, abs=0.001)
        assert result["bat_curr_max"] == pytest.approx(6.0)


class TestMotorFeatures:
    def test_no_rcou_messages(self):
        fe = _make_extractor()
        result = fe.motor_features()
        assert result["motor_available"] is False

    def test_basic_motor(self):
        msgs = [
            _make_msg(C1=1500, C2=1500, C3=1500, C4=1500),
            _make_msg(C1=1600, C2=1400, C3=1550, C4=1450),
        ]
        fe = _make_extractor({"RCOU": msgs})
        result = fe.motor_features()
        assert result["motor_available"] is True
        assert "motor_C1_mean" in result
        assert "motor_cross_channel_spread" in result

    def test_motor_spread_calculation(self):
        msgs = [
            _make_msg(C1=1900, C2=1100, C3=1500, C4=1500),
            _make_msg(C1=1900, C2=1100, C3=1500, C4=1500),
        ]
        fe = _make_extractor({"RCOU": msgs})
        result = fe.motor_features()
        expected_spread = float(np.std([1900, 1100, 1500, 1500]))
        assert result["motor_cross_channel_spread"] == pytest.approx(expected_spread)

    def test_saturation_detection(self):
        msgs = [
            _make_msg(C1=1950, C2=1500, C3=1500, C4=1500),
            _make_msg(C1=1500, C2=1500, C3=1500, C4=1500),
            _make_msg(C1=1500, C2=1960, C3=1980, C4=1500),
        ]
        fe = _make_extractor({"RCOU": msgs})
        result = fe.motor_features()
        assert result["motor_saturation_count"] == 2  # sample 0 and sample 2


class TestPowerFeatures:
    def test_no_powr_messages(self):
        fe = _make_extractor()
        result = fe.power_features()
        assert result["powr_available"] is False

    def test_brownout_risk(self):
        msgs = [
            _make_msg(Vcc=5.0),
            _make_msg(Vcc=4.8),
            _make_msg(Vcc=4.3),
            _make_msg(Vcc=4.4),
        ]
        fe = _make_extractor({"POWR": msgs})
        result = fe.power_features()
        assert result["powr_available"] is True
        assert result["powr_brownout_risk_count"] == 2  # 4.3 and 4.4


class TestPerformanceFeatures:
    def test_no_pm_messages(self):
        fe = _make_extractor()
        result = fe.performance_features()
        assert result["pm_available"] is False

    def test_scheduling_overrun(self):
        msgs = [
            _make_msg(NLon=0, MaxT=500, Load=30),
            _make_msg(NLon=3, MaxT=2000, Load=80),
            _make_msg(NLon=0, MaxT=600, Load=40),
        ]
        fe = _make_extractor({"PM": msgs})
        result = fe.performance_features()
        assert result["pm_available"] is True
        assert result["pm_overrun_count"] == 1
        assert result["pm_nlon_max"] == 3
        assert result["pm_max_t_max"] == pytest.approx(2000)


class TestControlFeatures:
    def test_no_ctun_messages(self):
        fe = _make_extractor()
        result = fe.control_features()
        assert result["ctun_available"] is False

    def test_basic_control(self):
        msgs = [
            _make_msg(ThO=0.5, Alt=10.0, DAlt=10.0),
            _make_msg(ThO=0.8, Alt=12.0, DAlt=10.0),
        ]
        fe = _make_extractor({"CTUN": msgs})
        result = fe.control_features()
        assert result["ctun_available"] is True
        assert result["ctun_tho_max"] == pytest.approx(0.8)
        assert result["ctun_alt_error_max"] == pytest.approx(2.0)


class TestRCFeatures:
    def test_no_rcin_messages(self):
        fe = _make_extractor()
        result = fe.rc_features()
        assert result["rcin_available"] is False

    def test_failsafe_detection(self):
        msgs = [
            _make_msg(C3=1500),
            _make_msg(C3=800),  # below 900 -> failsafe
            _make_msg(C3=850),
        ]
        fe = _make_extractor({"RCIN": msgs})
        result = fe.rc_features()
        assert result["rcin_available"] is True
        assert result["rcin_failsafe_count"] == 2


class TestErrorFeatures:
    def test_no_error_messages(self):
        fe = _make_extractor()
        result = fe.error_features()
        assert result["err_available"] is False

    def test_error_decoding(self):
        err_msgs = [
            _make_msg(Subsys=12, ECode=1),  # CrashCheck
            _make_msg(Subsys=6, ECode=2),   # EKF/InertialNav
            _make_msg(Subsys=99, ECode=0),  # Unknown
        ]
        ev_msgs = [_make_msg(Id=10), _make_msg(Id=15)]
        mode_msgs = [_make_msg(Mode=0), _make_msg(Mode=5)]
        fe = _make_extractor({"ERR": err_msgs, "EV": ev_msgs, "MODE": mode_msgs})
        result = fe.error_features()

        assert result["err_available"] is True
        assert result["err_total_count"] == 3
        assert result["err_crash_check_count"] == 1
        assert result["err_ekf_failsafe_count"] == 1
        assert result["err_decoded"][0]["subsys_name"] == "CrashCheck"
        assert result["err_decoded"][2]["subsys_name"] == "Unknown(99)"
        assert result["ev_count"] == 2
        assert result["mode_change_count"] == 2


class TestMagFeatures:
    def test_no_mag_messages(self):
        fe = _make_extractor()
        result = fe.mag_features()
        assert result["mag_available"] is False

    def test_basic_mag(self):
        msgs = [
            _make_msg(MagX=100, MagY=200, MagZ=300, OfsX=10, OfsY=20, OfsZ=30),
            _make_msg(MagX=110, MagY=210, MagZ=310, OfsX=15, OfsY=25, OfsZ=35),
        ]
        fe = _make_extractor({"MAG": msgs})
        result = fe.mag_features()
        assert result["mag_available"] is True
        assert result["mag_field_range"] > 0
        assert result["mag_interference_max"] > 0


class TestUtilities:
    def test_time_above_threshold_basic(self):
        ts = np.array([0.0, 1.0, 2.0, 3.0])
        vals = np.array([10.0, 20.0, 20.0, 5.0])
        result = FeatureExtractor._time_above_threshold(ts, vals, 15.0)
        # Samples 1 and 2 above 15 -> 1 second
        assert result == pytest.approx(1.0)

    def test_time_above_threshold_empty(self):
        ts = np.array([])
        vals = np.array([])
        assert FeatureExtractor._time_above_threshold(ts, vals, 10.0) == 0.0

    def test_time_above_threshold_single_sample(self):
        ts = np.array([1.0])
        vals = np.array([100.0])
        assert FeatureExtractor._time_above_threshold(ts, vals, 10.0) == 0.0

    def test_first_above_threshold_found(self):
        ts = np.array([0.0, 1.0, 2.0, 3.0])
        vals = np.array([5.0, 5.0, 15.0, 20.0])
        result = FeatureExtractor._first_above_threshold(ts, vals, 10.0)
        assert result == pytest.approx(2.0)

    def test_first_above_threshold_not_found(self):
        ts = np.array([0.0, 1.0, 2.0])
        vals = np.array([1.0, 2.0, 3.0])
        result = FeatureExtractor._first_above_threshold(ts, vals, 10.0)
        assert math.isnan(result)

    def test_decode_bitmask(self):
        flags = _decode_bitmask(0b00001010, EKF_FAULT_BITS)
        assert "bad_mag_y" in flags  # bit 1
        assert "bad_airspeed" in flags  # bit 3
        assert len(flags) == 2

    def test_decode_bitmask_zero(self):
        flags = _decode_bitmask(0, EKF_FAULT_BITS)
        assert flags == []


class TestExtractAll:
    def test_all_empty(self):
        fe = _make_extractor()
        result = fe.extract_all()
        assert result["vibe_available"] is False
        assert result["gps_available"] is False
        assert result["bat_available"] is False
        assert not any(k.startswith("_error_") for k in result)

    def test_catches_extractor_error(self):
        """If an extractor raises, extract_all records the error."""
        fe = _make_extractor()
        fe.vibration_features = MagicMock(side_effect=ValueError("boom"))
        result = fe.extract_all()
        assert "_error_vibration_features" in result
        assert "boom" in result["_error_vibration_features"]
        assert "gps_available" in result

    def test_partial_data(self):
        """extract_all works when some message types have data and others don't."""
        vibe_msgs = [
            _make_msg(TimeUS=0, VibeX=5, VibeY=5, VibeZ=5, Clip0=0, Clip1=0, Clip2=0),
        ]
        fe = _make_extractor({"VIBE": vibe_msgs})
        result = fe.extract_all()
        assert result["vibe_available"] is True
        assert result["gps_available"] is False
        assert not any(k.startswith("_error_") for k in result)

    def test_all_extractors_represented(self):
        """extract_all runs all 12 extractors."""
        fe = _make_extractor()
        result = fe.extract_all()
        expected_avail_keys = [
            "vibe_available", "ekf_available", "att_available",
            "gps_available", "bat_available", "motor_available",
            "powr_available", "pm_available", "ctun_available",
            "rcin_available", "err_available", "mag_available",
        ]
        for key in expected_avail_keys:
            assert key in result, f"Missing key: {key}"
