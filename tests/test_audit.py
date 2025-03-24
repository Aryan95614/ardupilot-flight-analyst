"""Parameter audit tests."""

import pytest

from flight_analyst.audit import (
    audit,
    check_arming,
    check_compass,
    check_ekf,
    check_failsafes,
    check_frame_config,
    check_gps,
    check_notch_filter,
    check_tuning,
)


def _good_params() -> dict[str, float]:
    return {
        "ATC_RAT_RLL_P": 0.22,
        "ATC_RAT_PIT_P": 0.20,
        "ATC_RAT_YAW_P": 0.25,
        "INS_HNTCH_ENABLE": 1,
        "BATT_FS_LOW_ACT": 2,
        "FS_THR_ENABLE": 1,
        "FS_EKF_ACTION": 1,
        "FRAME_CLASS": 1,
        "COMPASS_USE": 1,
        "GPS_TYPE": 1,
        "FS_EKF_THRESH": 0.8,
        "ARMING_CHECK": 1,
    }


def _bad_params() -> dict[str, float]:
    return {
        "ATC_RAT_RLL_P": 0.135,
        "ATC_RAT_PIT_P": 0.135,
        "ATC_RAT_YAW_P": 0.18,
        "INS_HNTCH_ENABLE": 0,
        "BATT_FS_LOW_ACT": 0,
        "FS_THR_ENABLE": 0,
        "FS_EKF_ACTION": 0,
        "FRAME_CLASS": 0,
        "COMPASS_USE": 0,
        "GPS_TYPE": 0,
        "FS_EKF_THRESH": 1.5,
        "ARMING_CHECK": 0,
    }


class TestCheckTuning:
    def test_no_warning_when_pids_changed(self):
        assert check_tuning(_good_params()) == []

    def test_warns_on_stock_defaults(self):
        warnings = check_tuning(_bad_params())
        assert len(warnings) == 3
        params_warned = {w["param"] for w in warnings}
        assert params_warned == {
            "ATC_RAT_RLL_P", "ATC_RAT_PIT_P", "ATC_RAT_YAW_P",
        }
        for w in warnings:
            assert w["severity"] == "high"

    def test_no_warning_when_only_one_pid_stock(self):
        params = _good_params()
        params["ATC_RAT_RLL_P"] = 0.135
        assert check_tuning(params) == []

    def test_no_warning_when_two_pids_stock(self):
        params = _good_params()
        params["ATC_RAT_RLL_P"] = 0.135
        params["ATC_RAT_PIT_P"] = 0.135
        assert check_tuning(params) == []


class TestCheckNotchFilter:
    def test_no_warning_when_enabled(self):
        assert check_notch_filter({"INS_HNTCH_ENABLE": 1}) == []

    def test_warns_when_disabled(self):
        warnings = check_notch_filter({"INS_HNTCH_ENABLE": 0})
        assert len(warnings) == 1
        assert warnings[0]["param"] == "INS_HNTCH_ENABLE"
        assert warnings[0]["severity"] == "medium"

    def test_no_warning_when_param_missing(self):
        """Missing param defaults to 1 (enabled) -- no warning."""
        assert check_notch_filter({}) == []


class TestCheckFailsafes:
    def test_no_warning_with_good_params(self):
        assert check_failsafes(_good_params()) == []

    def test_warns_on_disabled_battery_fs(self):
        params = _good_params()
        params["BATT_FS_LOW_ACT"] = 0
        warnings = check_failsafes(params)
        assert len(warnings) == 1
        assert warnings[0]["param"] == "BATT_FS_LOW_ACT"

    def test_warns_on_disabled_throttle_fs(self):
        params = _good_params()
        params["FS_THR_ENABLE"] = 0
        warnings = check_failsafes(params)
        assert len(warnings) == 1
        assert warnings[0]["param"] == "FS_THR_ENABLE"

    def test_warns_on_disabled_ekf_action(self):
        params = _good_params()
        params["FS_EKF_ACTION"] = 0
        warnings = check_failsafes(params)
        assert len(warnings) == 1
        assert warnings[0]["param"] == "FS_EKF_ACTION"

    def test_warns_on_all_three_disabled(self):
        params = _good_params()
        params["BATT_FS_LOW_ACT"] = 0
        params["FS_THR_ENABLE"] = 0
        params["FS_EKF_ACTION"] = 0
        assert len(check_failsafes(params)) == 3


class TestCheckFrameConfig:
    def test_no_warning_when_set(self):
        assert check_frame_config({"FRAME_CLASS": 1}) == []

    def test_warns_when_zero(self):
        warnings = check_frame_config({"FRAME_CLASS": 0})
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "high"

    def test_no_warning_when_missing(self):
        assert check_frame_config({}) == []


class TestCheckCompass:
    def test_no_warning_when_enabled(self):
        assert check_compass({"COMPASS_USE": 1}) == []

    def test_warns_when_disabled(self):
        warnings = check_compass({"COMPASS_USE": 0})
        assert len(warnings) == 1
        assert warnings[0]["param"] == "COMPASS_USE"


class TestCheckGPS:
    def test_no_warning_when_set(self):
        assert check_gps({"GPS_TYPE": 1}) == []

    def test_warns_when_disabled(self):
        warnings = check_gps({"GPS_TYPE": 0})
        assert len(warnings) == 1
        assert warnings[0]["param"] == "GPS_TYPE"

    def test_no_warning_when_missing(self):
        assert check_gps({}) == []


class TestCheckEKF:
    def test_no_warning_at_default(self):
        assert check_ekf({"FS_EKF_THRESH": 0.8}) == []

    def test_warns_when_threshold_too_high(self):
        warnings = check_ekf({"FS_EKF_THRESH": 1.5})
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "medium"

    def test_no_warning_at_boundary(self):
        assert check_ekf({"FS_EKF_THRESH": 1.0}) == []

    def test_warns_just_above_boundary(self):
        assert len(check_ekf({"FS_EKF_THRESH": 1.01})) == 1


class TestCheckArming:
    def test_no_warning_when_enabled(self):
        assert check_arming({"ARMING_CHECK": 1}) == []

    def test_warns_when_disabled(self):
        warnings = check_arming({"ARMING_CHECK": 0})
        assert len(warnings) == 1
        assert warnings[0]["param"] == "ARMING_CHECK"


class TestAudit:
    def test_no_warnings_on_good_params(self):
        assert audit(_good_params()) == []

    def test_all_warnings_on_bad_params(self):
        warnings = audit(_bad_params())
        # 3 PIDs + notch + 3 failsafes + frame + compass + gps + ekf + arming
        assert len(warnings) == 12

    def test_warning_dict_structure(self):
        warnings = audit(_bad_params())
        required_keys = {
            "severity", "param", "value", "issue",
            "recommendation", "wiki_url",
        }
        for w in warnings:
            assert required_keys.issubset(w.keys())

    def test_every_warning_has_wiki_url(self):
        for w in audit(_bad_params()):
            assert w["wiki_url"].startswith("https://")

    def test_empty_params_no_crash(self):
        """An empty dict should not raise; defaults are safe."""
        result = audit({})
        assert isinstance(result, list)
