"""LogParser tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flight_analyst.parser import LogParser


def _make_parser() -> LogParser:
    with patch("flight_analyst.parser.mavutil.mavlink_connection") as mock_conn:
        mlog = MagicMock()
        mlog.recv_match.return_value = None
        mock_conn.return_value = mlog
        return LogParser("/fake/log.bin")


def _ts(base: float, n: int, step: float = 1.0) -> list[float]:
    return [base + i * step for i in range(n)]


class TestMessageTypes:
    def test_get_message_types_empty(self):
        p = _make_parser()
        assert p.get_message_types() == []

    def test_get_message_types_sorted(self):
        p = _make_parser()
        p._cache = {"GPS": [{}], "ATT": [{}], "VIBE": [{}]}
        assert p.get_message_types() == ["ATT", "GPS", "VIBE"]

    def test_has_message_type_true(self):
        p = _make_parser()
        p._cache = {"ATT": [{"Roll": 1.0}]}
        assert p.has_message_type("ATT") is True

    def test_has_message_type_false(self):
        p = _make_parser()
        assert p.has_message_type("ATT") is False

    def test_has_message_type_empty_list(self):
        p = _make_parser()
        p._cache = {"ATT": []}
        assert p.has_message_type("ATT") is False

    def test_get_messages_returns_copy(self):
        p = _make_parser()
        p._cache = {"ATT": [{"Roll": 1.0}]}
        msgs = p.get_messages("ATT")
        msgs.append({"Roll": 99.0})
        assert len(p.get_messages("ATT")) == 1  # original unchanged

    def test_get_messages_missing_type(self):
        p = _make_parser()
        assert p.get_messages("NONEXISTENT") == []


class TestTimeSeries:
    def test_time_series_with_timeus(self):
        p = _make_parser()
        p._cache = {
            "ATT": [
                {"TimeUS": 1_000_000, "Roll": 1.5},
                {"TimeUS": 2_000_000, "Roll": 2.5},
                {"TimeUS": 3_000_000, "Roll": 3.5},
            ]
        }
        t, v = p.get_time_series("ATT", "Roll")
        np.testing.assert_allclose(t, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(v, [1.5, 2.5, 3.5])

    def test_time_series_with_timestamp_fallback(self):
        p = _make_parser()
        p._cache = {
            "ATT": [
                {"_timestamp": 100.0, "Roll": 10.0},
                {"_timestamp": 101.0, "Roll": 20.0},
            ]
        }
        t, v = p.get_time_series("ATT", "Roll")
        np.testing.assert_allclose(t, [100.0, 101.0])
        np.testing.assert_allclose(v, [10.0, 20.0])

    def test_time_series_missing_type_raises(self):
        p = _make_parser()
        with pytest.raises(KeyError, match="No messages"):
            p.get_time_series("ATT", "Roll")

    def test_time_series_missing_field_raises(self):
        p = _make_parser()
        p._cache = {"ATT": [{"TimeUS": 0, "Roll": 1.0}]}
        with pytest.raises(ValueError, match="Field.*not found"):
            p.get_time_series("ATT", "NoSuchField")


class TestEKFDetection:
    def test_ekf3_detected_via_xkf4(self):
        p = _make_parser()
        p._cache = {
            "XKF4": [
                {"TimeUS": 1_000_000, "SV": 10, "SP": 1, "SH": 1,
                 "SM": 1, "SVT": 1, "OFN": 0.0, "OFE": 0.0,
                 "FS": 0, "TS": 0, "SS": 0}
            ]
        }
        assert p.get_ekf_version() == 3
        assert p.get_ekf_health_msg_type() == "XKF4"

    def test_ekf2_detected_via_nkf4(self):
        p = _make_parser()
        p._cache = {
            "NKF4": [
                {"TimeUS": 1_000_000, "SV": 5, "SP": 1, "SH": 1,
                 "SM": 1, "SVT": 1, "OFN": 0.0, "OFE": 0.0,
                 "FS": 0, "TS": 0, "SS": 0}
            ]
        }
        assert p.get_ekf_version() == 2
        assert p.get_ekf_health_msg_type() == "NKF4"

    def test_ekf2_detected_via_ekf4(self):
        p = _make_parser()
        p._cache = {
            "EKF4": [
                {"TimeUS": 1_000_000, "SV": 5, "SP": 1, "SH": 1}
            ]
        }
        assert p.get_ekf_version() == 2
        assert p.get_ekf_health_msg_type() == "EKF4"

    def test_ekf_version_none_when_absent(self):
        p = _make_parser()
        assert p.get_ekf_version() is None
        assert p.get_ekf_health_msg_type() is None

    def test_ekf3_takes_priority_over_ekf2(self):
        """If both XKF4 and NKF4 are present, EKF3 wins."""
        p = _make_parser()
        p._cache = {
            "XKF4": [{"TimeUS": 0}],
            "NKF4": [{"TimeUS": 0}],
        }
        assert p.get_ekf_version() == 3


class TestVehicleType:
    def test_vehicle_copter_from_msg(self):
        p = _make_parser()
        p._cache = {
            "MSG": [{"Message": "ArduCopter V4.5.1 (abc123)"}]
        }
        assert p.get_vehicle_type() == "Copter"

    def test_vehicle_plane_from_msg(self):
        p = _make_parser()
        p._cache = {
            "MSG": [{"Message": "ArduPlane V4.4.0"}]
        }
        assert p.get_vehicle_type() == "Plane"

    def test_vehicle_rover_from_msg(self):
        p = _make_parser()
        p._cache = {
            "MSG": [{"Message": "APMrover2 V4.2.0"}]
        }
        assert p.get_vehicle_type() == "Rover"

    def test_vehicle_sub_from_msg(self):
        p = _make_parser()
        p._cache = {"MSG": [{"Message": "ArduSub V4.1.0"}]}
        assert p.get_vehicle_type() == "Sub"

    def test_vehicle_from_frame_class(self):
        p = _make_parser()
        p._cache = {
            "PARM": [{"Name": "FRAME_CLASS", "Value": 1}]
        }
        assert p.get_vehicle_type() == "Copter"

    def test_vehicle_type_none_when_unknown(self):
        p = _make_parser()
        assert p.get_vehicle_type() is None


class TestParams:
    def test_get_params(self):
        p = _make_parser()
        p._cache = {
            "PARM": [
                {"Name": "ARMING_CHECK", "Value": 1},
                {"Name": "INS_GYRO_FILTER", "Value": 20},
            ]
        }
        params = p.get_params()
        assert params == {"ARMING_CHECK": 1.0, "INS_GYRO_FILTER": 20.0}

    def test_get_params_empty(self):
        p = _make_parser()
        assert p.get_params() == {}


class TestModesErrorsEvents:
    def test_get_flight_modes(self):
        p = _make_parser()
        p._cache = {
            "MODE": [
                {"TimeUS": 0, "Mode": 0, "ModeNum": 0, "Rsn": 0},
                {"TimeUS": 5_000_000, "Mode": 5, "ModeNum": 5, "Rsn": 1},
            ]
        }
        modes = p.get_flight_modes()
        assert len(modes) == 2
        assert modes[1]["Mode"] == 5

    def test_get_errors(self):
        p = _make_parser()
        p._cache = {
            "ERR": [
                {"TimeUS": 0, "Subsys": 18, "ECode": 2},
            ]
        }
        errs = p.get_errors()
        assert len(errs) == 1
        assert errs[0]["Subsys"] == 18

    def test_get_events(self):
        p = _make_parser()
        p._cache = {
            "EV": [
                {"TimeUS": 0, "Id": 10},
                {"TimeUS": 1_000_000, "Id": 15},
            ]
        }
        evs = p.get_events()
        assert len(evs) == 2


class TestFirmwareVersion:
    def test_version_from_ver_message(self):
        p = _make_parser()
        p._cache = {
            "VER": [{"APJ": "4.5.1", "GH": "abc123"}]
        }
        assert p.get_firmware_version() == "4.5.1"

    def test_version_from_msg_fallback(self):
        p = _make_parser()
        p._cache = {
            "MSG": [{"Message": "ArduCopter V4.3.7 (abcd1234)"}]
        }
        assert p.get_firmware_version() == "V4.3.7"

    def test_version_none_when_absent(self):
        p = _make_parser()
        assert p.get_firmware_version() is None


class TestFlightDuration:
    def test_duration_from_timestamps(self):
        p = _make_parser()
        p._cache = {
            "ATT": [
                {"_timestamp": 100.0, "Roll": 0},
                {"_timestamp": 200.0, "Roll": 0},
            ],
            "GPS": [
                {"_timestamp": 150.0, "Lat": 0},
                {"_timestamp": 250.0, "Lat": 0},
            ],
        }
        assert p.get_flight_duration() == pytest.approx(150.0)

    def test_duration_zero_without_timestamps(self):
        p = _make_parser()
        p._cache = {"ATT": [{"Roll": 0}]}
        assert p.get_flight_duration() == 0.0


class TestVariousMessageTypes:
    def test_vibe_messages(self):
        p = _make_parser()
        p._cache = {
            "VIBE": [
                {"TimeUS": 1_000_000, "VibeX": 0.5, "VibeY": 0.3, "VibeZ": 0.8, "Clip0": 0},
                {"TimeUS": 2_000_000, "VibeX": 0.6, "VibeY": 0.4, "VibeZ": 0.9, "Clip0": 1},
            ]
        }
        t, v = p.get_time_series("VIBE", "VibeZ")
        np.testing.assert_allclose(v, [0.8, 0.9])

    def test_gps_messages(self):
        p = _make_parser()
        p._cache = {
            "GPS": [
                {"TimeUS": 0, "Lat": 37.0, "Lng": -122.0, "NSats": 12},
            ]
        }
        assert p.get_messages("GPS")[0]["NSats"] == 12

    def test_bat_messages(self):
        p = _make_parser()
        p._cache = {
            "BAT": [
                {"TimeUS": 0, "Volt": 16.2, "Curr": 10.5},
                {"TimeUS": 1_000_000, "Volt": 15.8, "Curr": 12.0},
            ]
        }
        t, v = p.get_time_series("BAT", "Volt")
        np.testing.assert_allclose(v, [16.2, 15.8])

    def test_rcou_messages(self):
        p = _make_parser()
        p._cache = {
            "RCOU": [
                {"TimeUS": 0, "C1": 1500, "C2": 1500, "C3": 1100, "C4": 1500},
            ]
        }
        assert p.has_message_type("RCOU")
        assert p.get_messages("RCOU")[0]["C3"] == 1100

    def test_powr_messages(self):
        p = _make_parser()
        p._cache = {
            "POWR": [
                {"TimeUS": 0, "Vcc": 5.02, "VServo": 5.1},
            ]
        }
        t, v = p.get_time_series("POWR", "Vcc")
        np.testing.assert_allclose(v, [5.02])

    def test_pm_messages(self):
        p = _make_parser()
        p._cache = {
            "PM": [
                {"TimeUS": 0, "NLon": 5, "NLoop": 400, "MaxT": 1200},
            ]
        }
        assert p.get_messages("PM")[0]["MaxT"] == 1200


class TestPreload:
    def test_preload_skips_bad_data(self):
        """BAD_DATA messages must not appear in the cache."""
        with patch("flight_analyst.parser.mavutil.mavlink_connection") as mock_conn:
            bad_msg = MagicMock()
            bad_msg.get_type.return_value = "BAD_DATA"

            good_msg = MagicMock()
            good_msg.get_type.return_value = "ATT"
            good_msg.to_dict.return_value = {"Roll": 1.0}
            good_msg._timestamp = 100.0

            mlog = MagicMock()
            mlog.recv_match.side_effect = [bad_msg, good_msg, None]
            mock_conn.return_value = mlog

            p = LogParser("/fake/log.bin")

        assert "BAD_DATA" not in p._cache
        assert p.has_message_type("ATT")
        assert p.get_messages("ATT")[0]["_timestamp"] == 100.0
