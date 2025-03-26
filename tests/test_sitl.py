"""SITL fault injection tests (all MAVLink connections mocked)."""

import os
import tempfile
from unittest import mock

import pytest

from flight_analyst.sitl.inject import SITLInjector
from flight_analyst.sitl.scenarios import (
    SCENARIOS,
    _REQUIRED_KEYS,
    _REQUIRED_SETUP_KEYS,
    get_scenario,
    list_scenarios,
    validate_scenario,
)
from flight_analyst.sitl.generate_dataset import (
    _FAULT_METHOD_MAP,
    _apply_fault,
    run_scenario,
    generate_dataset,
)


def _make_heartbeat(custom_mode: int = 5, armed: bool = False):
    hb = mock.MagicMock()
    hb.get_type.return_value = "HEARTBEAT"
    hb.custom_mode = custom_mode
    base = 0
    if armed:
        from pymavlink import mavutil
        base |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    hb.base_mode = base
    return hb


def _make_param_value(name: str, value: float):
    msg = mock.MagicMock()
    msg.get_type.return_value = "PARAM_VALUE"
    msg.param_id = name
    msg.param_value = value
    return msg


def _make_global_pos(relative_alt_mm: int):
    msg = mock.MagicMock()
    msg.get_type.return_value = "GLOBAL_POSITION_INT"
    msg.relative_alt = relative_alt_mm
    return msg


def _make_injector() -> SITLInjector:
    inj = SITLInjector("tcp:127.0.0.1:5760")
    inj.mav = mock.MagicMock()
    inj.mav.target_system = 1
    inj.mav.target_component = 1
    return inj


def _auto_ack_side_effect(injector: SITLInjector):
    def _side_effect(**kw):
        calls = injector.mav.mav.param_set_send.call_args_list
        if calls:
            last_name = calls[-1][0][2]
            if isinstance(last_name, bytes):
                last_name = last_name.decode("utf-8")
            return _make_param_value(last_name, 0)
        return _make_param_value("x", 0)
    return _side_effect


class TestScenarioDefinitions:
    def test_scenario_count(self):
        assert len(SCENARIOS) == 10

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_required_top_level_keys(self, name):
        scenario = SCENARIOS[name]
        assert _REQUIRED_KEYS.issubset(scenario.keys()), (
            f"Scenario '{name}' missing keys: "
            f"{_REQUIRED_KEYS - scenario.keys()}"
        )

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_required_setup_keys(self, name):
        setup = SCENARIOS[name]["setup"]
        assert _REQUIRED_SETUP_KEYS.issubset(setup.keys())

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_fault_structure(self, name):
        fault = SCENARIOS[name]["fault"]
        if fault is not None:
            assert isinstance(fault, dict)
            assert "type" in fault

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_validate_returns_true(self, name):
        assert validate_scenario(SCENARIOS[name]) is True

    def test_normal_flight_has_no_fault(self):
        assert SCENARIOS["normal_flight"]["fault"] is None
        assert SCENARIOS["normal_flight"]["observe_time"] == 0


class TestScenarioHelpers:
    def test_get_scenario_valid(self):
        s = get_scenario("motor_failure_hover")
        assert s["label"] == "motor_failure"

    def test_get_scenario_invalid(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("nonexistent")

    def test_list_scenarios(self):
        names = list_scenarios()
        assert isinstance(names, list)
        assert "motor_failure_hover" in names
        assert len(names) == 10

    def test_validate_bad_input(self):
        assert validate_scenario("not a dict") is False

    def test_validate_missing_keys(self):
        assert validate_scenario({"label": "x"}) is False

    def test_validate_bad_fault(self):
        bad = {
            "label": "x",
            "description": "x",
            "setup": {"target_alt": 1, "hover_time": 1, "mode": "LOITER"},
            "fault": {"no_type_key": True},
            "observe_time": 5,
        }
        assert validate_scenario(bad) is False


class TestInjectorParams:
    def test_set_param_sends_correct_name_and_value(self):
        inj = _make_injector()
        inj.mav.recv_match.return_value = _make_param_value("SIM_ENGINE_FAIL", 1)
        inj.set_param("SIM_ENGINE_FAIL", 1)

        call_args = inj.mav.mav.param_set_send.call_args
        assert call_args[0][2] == b"SIM_ENGINE_FAIL"
        assert call_args[0][3] == 1.0

    def test_get_param_returns_value(self):
        inj = _make_injector()
        inj.mav.recv_match.return_value = _make_param_value("SIM_WIND_SPD", 7.5)
        val = inj.get_param("SIM_WIND_SPD")
        assert val == 7.5

    def test_set_param_raises_without_connection(self):
        inj = SITLInjector()
        with pytest.raises(RuntimeError, match="Not connected"):
            inj.set_param("X", 0)

    def test_get_param_raises_without_connection(self):
        inj = SITLInjector()
        with pytest.raises(RuntimeError, match="Not connected"):
            inj.get_param("X")


class TestFaultInjection:
    def _setup_inj(self):
        inj = _make_injector()
        inj.mav.recv_match.side_effect = _auto_ack_side_effect(inj)
        return inj

    def test_inject_motor_failure_params(self):
        inj = self._setup_inj()
        inj.inject_motor_failure(motor=3, efficiency=0.0)
        calls = [c[0][2] for c in inj.mav.mav.param_set_send.call_args_list]
        assert b"SIM_ENGINE_FAIL" in calls
        assert b"SIM_ENGINE_MUL" in calls
        # motor 3 -> bitmask 0b100 = 4
        fail_call = [
            c for c in inj.mav.mav.param_set_send.call_args_list
            if c[0][2] == b"SIM_ENGINE_FAIL"
        ][0]
        assert fail_call[0][3] == 4.0

    def test_inject_gps_glitch_params(self):
        inj = self._setup_inj()
        inj.inject_gps_glitch(lat_offset=0.002, lon_offset=0.003)
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_GPS1_GLTCH_X"] == 0.002
        assert calls[b"SIM_GPS1_GLTCH_Y"] == 0.003

    def test_inject_gps_loss(self):
        inj = self._setup_inj()
        inj.inject_gps_loss()
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_GPS1_ENABLE"] == 0.0

    def test_inject_battery_drop(self):
        inj = self._setup_inj()
        inj.inject_battery_drop(voltage=9.5)
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_BATT_VOLTAGE"] == 9.5

    def test_inject_vibration_all_axes(self):
        inj = self._setup_inj()
        inj.inject_vibration(freq=200, noise=15)
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_VIB_FREQ_X"] == 200
        assert calls[b"SIM_VIB_FREQ_Y"] == 200
        assert calls[b"SIM_VIB_FREQ_Z"] == 200
        assert calls[b"SIM_ACC1_RND"] == 15

    def test_inject_rc_failsafe(self):
        inj = self._setup_inj()
        inj.inject_rc_failsafe()
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_RC_FAIL"] == 1.0

    def test_inject_compass_interference(self):
        inj = self._setup_inj()
        inj.inject_compass_interference(offset=400)
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_MAG1_OFS_X"] == 400
        assert calls[b"SIM_MAG1_OFS_Y"] == 400
        assert calls[b"SIM_MAG1_OFS_Z"] == 400

    def test_inject_imu_bias(self):
        inj = self._setup_inj()
        inj.inject_imu_bias(accel_bias=3.0)
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_ACC1_BIAS_X"] == 3.0
        assert calls[b"SIM_ACC1_BIAS_Z"] == 3.0

    def test_inject_wind(self):
        inj = self._setup_inj()
        inj.inject_wind(speed=12, turbulence=3)
        calls = {c[0][2]: c[0][3] for c in inj.mav.mav.param_set_send.call_args_list}
        assert calls[b"SIM_WIND_SPD"] == 12
        assert calls[b"SIM_WIND_TURB"] == 3


class TestClearAllFaults:
    def test_resets_all_defaults(self):
        inj = _make_injector()
        inj.mav.recv_match.side_effect = _auto_ack_side_effect(inj)
        inj.clear_all_faults()

        sent = {
            c[0][2]: c[0][3]
            for c in inj.mav.mav.param_set_send.call_args_list
        }
        for name, default in SITLInjector.FAULT_DEFAULTS.items():
            assert sent[name.encode("utf-8")] == float(default), (
                f"{name} should be reset to {default}"
            )

    def test_clear_all_covers_every_default_key(self):
        inj = _make_injector()
        inj.mav.recv_match.side_effect = _auto_ack_side_effect(inj)
        inj.clear_all_faults()
        assert (
            inj.mav.mav.param_set_send.call_count
            == len(SITLInjector.FAULT_DEFAULTS)
        )


class TestGetLatestLog:
    def test_returns_newest(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["00000001.BIN", "00000002.BIN", "00000003.BIN"]:
                open(os.path.join(td, name), "w").close()
            result = SITLInjector.get_latest_log(log_dir=td)
            assert result is not None
            assert result.endswith("00000003.BIN")

    def test_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            assert SITLInjector.get_latest_log(log_dir=td) is None

    def test_ignores_non_bin_files(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "notes.txt"), "w").close()
            assert SITLInjector.get_latest_log(log_dir=td) is None


class TestApplyFault:
    def test_dispatches_motor_failure(self):
        inj = _make_injector()
        inj.inject_motor_failure = mock.MagicMock()
        _apply_fault(inj, {"type": "motor_failure", "motor": 1, "efficiency": 0.0})
        inj.inject_motor_failure.assert_called_once_with(motor=1, efficiency=0.0)

    def test_dispatches_gps_glitch(self):
        inj = _make_injector()
        inj.inject_gps_glitch = mock.MagicMock()
        _apply_fault(inj, {"type": "gps_glitch", "lat_offset": 0.1, "lon_offset": 0.2})
        inj.inject_gps_glitch.assert_called_once_with(lat_offset=0.1, lon_offset=0.2)

    def test_unknown_fault_raises(self):
        inj = _make_injector()
        with pytest.raises(ValueError, match="Unknown fault type"):
            _apply_fault(inj, {"type": "warp_drive"})

    def test_all_scenario_faults_in_method_map(self):
        for name, scenario in SCENARIOS.items():
            fault = scenario["fault"]
            if fault is not None:
                assert fault["type"] in _FAULT_METHOD_MAP, (
                    f"Scenario '{name}' uses fault type '{fault['type']}' "
                    f"not in _FAULT_METHOD_MAP"
                )


class TestModeAndArm:
    def test_change_mode_unknown_raises(self):
        inj = _make_injector()
        with pytest.raises(ValueError, match="Unknown mode"):
            inj.change_mode("WARP_SPEED")

    def test_arm_raises_without_connection(self):
        inj = SITLInjector()
        with pytest.raises(RuntimeError, match="Not connected"):
            inj.arm()

    def test_disarm_force_passes_magic_p2(self):
        inj = _make_injector()
        hb = _make_heartbeat(armed=False)
        inj.mav.recv_match.return_value = hb
        inj.disarm(force=True)
        call_args = inj.mav.mav.command_long_send.call_args[0]
        assert call_args[4] == 0      # p1 = 0 (disarm)
        assert call_args[5] == 21196  # p2 = force magic


class TestConnection:
    @mock.patch("flight_analyst.sitl.inject.mavutil")
    def test_connect_waits_heartbeat(self, mock_mavutil):
        inj = SITLInjector("tcp:127.0.0.1:5760")
        inj.connect()
        mock_mavutil.mavlink_connection.assert_called_once_with(
            "tcp:127.0.0.1:5760", source_system=250,
        )
        mock_mavutil.mavlink_connection.return_value.wait_heartbeat.assert_called_once()
