"""DiagnosticClassifier tests."""

import pytest

from flight_analyst.classifier import (
    CAUSAL_CHAINS,
    FAILURE_CLASSES,
    DiagnosticClassifier,
    _confidence_above,
    _confidence_below,
)


def _base_features(**overrides) -> dict:
    defaults = {
        "vibe_x_max": 10.0,
        "vibe_z_max": 10.0,
        "clips": 0,
        "vibe_onset_time": 0.0,
        "ekf_sv_max": 0.5,
        "ekf_fault_count": 0,
        "ekf_sv_onset_time": 0.0,
        "mag_field_range": 100.0,
        "mag_interference_likely": False,
        "mag_onset_time": 0.0,
        "gps_hdop_max": 1.5,
        "gps_nsats_min": 12,
        "err_has_gps_glitch": False,
        "gps_onset_time": 0.0,
        "motor_spread_max": 100.0,
        "motor_spread_onset_time": 0.0,
        "motor_saturation_high_count": 0,
        "motor_saturation_low_count": 0,
        "motor_failure_onset_time": 0.0,
        "bat_volt_min": 14.8,
        "bat_max_drop_rate": -0.3,
        "power_onset_time": 0.0,
        "rc_likely_failsafe": False,
        "err_subsystem_2": False,
        "err_subsystem_5": False,
        "rc_failsafe_onset_time": 0.0,
        "motor_saturation_high_time": 0.0,
        "thrust_loss_onset_time": 0.0,
        "stock_pids_detected": False,
        "attitude_error_max": 3.0,
        "tuning_onset_time": 0.0,
        "mechanical_failure_detected": False,
        "mechanical_failure_confidence": 0.7,
        "mechanical_failure_onset_time": 0.0,
        "esc_desync_detected": False,
        "esc_desync_confidence": 0.7,
        "esc_desync_onset_time": 0.0,
        "powr_vcc_min": 5.0,
        "brownout_onset_time": 0.0,
        "pm_scheduling_overrun": False,
        "pm_nlon_total": 2,
        "scheduling_onset_time": 0.0,
    }
    defaults.update(overrides)
    return defaults


class TestFailureClasses:
    def test_count(self):
        assert len(FAILURE_CLASSES) == 15

    def test_normal_in_list(self):
        assert "normal" in FAILURE_CLASSES

    def test_all_unique(self):
        assert len(FAILURE_CLASSES) == len(set(FAILURE_CLASSES))


class TestVibrationDetection:
    def test_high_vibe_x(self):
        feats = _base_features(vibe_x_max=60.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "vibration_high" in classes

    def test_high_vibe_z(self):
        feats = _base_features(vibe_z_max=45.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "vibration_high" in classes

    def test_low_vibe_no_detection(self):
        feats = _base_features(vibe_x_max=20.0, vibe_z_max=20.0, clips=0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "vibration_high" not in classes

    def test_clips_trigger(self):
        feats = _base_features(clips=5)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "vibration_high" in classes


class TestEKFFailure:
    def test_high_sv(self):
        feats = _base_features(ekf_sv_max=1.5)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "ekf_failure" in classes

    def test_fault_count(self):
        feats = _base_features(ekf_fault_count=3)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "ekf_failure" in classes

    def test_normal_sv_no_detection(self):
        feats = _base_features(ekf_sv_max=0.8, ekf_fault_count=0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "ekf_failure" not in classes


class TestCompassInterference:
    def test_high_mag_range(self):
        feats = _base_features(mag_field_range=350.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "compass_interference" in classes

    def test_mag_interference_flag(self):
        feats = _base_features(mag_interference_likely=True)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "compass_interference" in classes

    def test_no_compass_issue(self):
        feats = _base_features(mag_field_range=100.0, mag_interference_likely=False)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "compass_interference" not in classes


class TestGPSGlitch:
    def test_high_hdop(self):
        feats = _base_features(gps_hdop_max=5.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "gps_glitch" in classes

    def test_low_nsats(self):
        feats = _base_features(gps_nsats_min=3)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "gps_glitch" in classes

    def test_err_gps_flag(self):
        feats = _base_features(err_has_gps_glitch=True)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "gps_glitch" in classes

    def test_good_gps(self):
        feats = _base_features(gps_hdop_max=1.0, gps_nsats_min=14)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "gps_glitch" not in classes


class TestMotorImbalance:
    def test_high_spread(self):
        feats = _base_features(motor_spread_max=500.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "motor_imbalance" in classes

    def test_low_spread(self):
        feats = _base_features(motor_spread_max=100.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "motor_imbalance" not in classes


class TestMotorFailure:
    def test_simultaneous_saturation(self):
        feats = _base_features(
            motor_saturation_high_count=5,
            motor_saturation_low_count=3,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "motor_failure" in classes

    def test_only_high_saturation(self):
        feats = _base_features(
            motor_saturation_high_count=5,
            motor_saturation_low_count=0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "motor_failure" not in classes


class TestPowerIssue:
    def test_low_voltage(self):
        feats = _base_features(bat_volt_min=9.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "power_issue" in classes

    def test_high_drop_rate(self):
        feats = _base_features(bat_max_drop_rate=-2.5)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "power_issue" in classes

    def test_healthy_battery(self):
        feats = _base_features(bat_volt_min=14.0, bat_max_drop_rate=-0.2)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "power_issue" not in classes


class TestRCFailsafe:
    def test_rc_failsafe_flag(self):
        feats = _base_features(rc_likely_failsafe=True)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "rc_failsafe" in classes

    def test_err_subsystem_2(self):
        feats = _base_features(err_subsystem_2=True)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "rc_failsafe" in classes

    def test_no_rc_issue(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "rc_failsafe" not in classes


class TestThrustLoss:
    def test_sustained_saturation(self):
        feats = _base_features(motor_saturation_high_time=5.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "thrust_loss" in classes

    def test_brief_saturation(self):
        feats = _base_features(motor_saturation_high_time=2.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "thrust_loss" not in classes


class TestBrownout:
    def test_low_vcc(self):
        feats = _base_features(powr_vcc_min=4.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "brownout" in classes

    def test_normal_vcc(self):
        feats = _base_features(powr_vcc_min=5.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "brownout" not in classes


class TestSchedulingOverrun:
    def test_overrun_flag(self):
        feats = _base_features(pm_scheduling_overrun=True)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "scheduling_overrun" in classes

    def test_high_nlon(self):
        feats = _base_features(pm_nlon_total=25)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "scheduling_overrun" in classes

    def test_low_nlon(self):
        feats = _base_features(pm_nlon_total=5)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "scheduling_overrun" not in classes


class TestNormalFlight:
    def test_clean_features_yield_normal(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        assert len(dets) == 1
        assert dets[0]["failure_class"] == "normal"
        assert dets[0]["confidence"] == 1.0

    def test_normal_evidence(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        assert "No anomalies" in dets[0]["evidence"]


class TestConfidence:
    def test_confidence_above_basic(self):
        assert _confidence_above(60, 30, 30) == 1.0

    def test_confidence_above_partial(self):
        c = _confidence_above(45, 30, 30)
        assert 0.49 < c < 0.51

    def test_confidence_above_below_threshold(self):
        assert _confidence_above(20, 30) == 0.0

    def test_confidence_below_basic(self):
        assert _confidence_below(0, 4.5, 4.5) == 1.0

    def test_confidence_below_partial(self):
        c = _confidence_below(3.0, 4.5, 4.5)
        assert 0.3 < c < 0.34

    def test_confidence_below_above_threshold(self):
        assert _confidence_below(5.0, 4.5) == 0.0

    def test_higher_severity_means_higher_confidence_vibration(self):
        feats_mild = _base_features(vibe_x_max=35.0)
        feats_severe = _base_features(vibe_x_max=60.0)
        dc_mild = DiagnosticClassifier(feats_mild)
        dc_severe = DiagnosticClassifier(feats_severe)
        dets_mild = dc_mild.classify_rule_based()
        dets_severe = dc_severe.classify_rule_based()
        vibe_mild = next(d for d in dets_mild if d["failure_class"] == "vibration_high")
        vibe_severe = next(d for d in dets_severe if d["failure_class"] == "vibration_high")
        assert vibe_severe["confidence"] > vibe_mild["confidence"]

    def test_higher_severity_means_higher_confidence_power(self):
        feats_mild = _base_features(bat_volt_min=10.0)
        feats_severe = _base_features(bat_volt_min=7.0)
        dc_mild = DiagnosticClassifier(feats_mild)
        dc_severe = DiagnosticClassifier(feats_severe)
        dets_mild = dc_mild.classify_rule_based()
        dets_severe = dc_severe.classify_rule_based()
        pwr_mild = next(d for d in dets_mild if d["failure_class"] == "power_issue")
        pwr_severe = next(d for d in dets_severe if d["failure_class"] == "power_issue")
        assert pwr_severe["confidence"] > pwr_mild["confidence"]


class TestSorting:
    def test_results_sorted_by_confidence_descending(self):
        feats = _base_features(
            vibe_x_max=35.0,
            ekf_sv_max=2.5,
            gps_hdop_max=4.0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        confs = [d["confidence"] for d in dets]
        assert confs == sorted(confs, reverse=True)


class TestCausalArbiter:
    def test_vibration_before_ekf(self):
        """Vibration at t=5 before EKF at t=10 -> vibration is root cause."""
        feats = _base_features(
            vibe_x_max=50.0, vibe_onset_time=5.0,
            ekf_sv_max=1.5, ekf_sv_onset_time=10.0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        dets = dc.causal_arbiter(dets)

        vibe = next(d for d in dets if d["failure_class"] == "vibration_high")
        ekf = next(d for d in dets if d["failure_class"] == "ekf_failure")
        assert vibe["is_root_cause"] is True
        assert ekf["is_downstream"] is True
        assert ekf["caused_by"] == "vibration_high"

    def test_motor_failure_before_compass(self):
        """Motor failure at t=2 before compass at t=8 -> motor is root cause."""
        feats = _base_features(
            motor_saturation_high_count=3,
            motor_saturation_low_count=2,
            motor_failure_onset_time=2.0,
            mag_field_range=400.0,
            mag_onset_time=8.0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        dets = dc.causal_arbiter(dets)

        motor = next(d for d in dets if d["failure_class"] == "motor_failure")
        compass = next(d for d in dets if d["failure_class"] == "compass_interference")
        assert motor["is_root_cause"] is True
        assert compass["is_downstream"] is True
        assert compass["caused_by"] == "motor_failure"

    def test_single_detection_is_root(self):
        feats = _base_features(vibe_x_max=50.0, vibe_onset_time=3.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        dets = dc.causal_arbiter(dets)
        assert len([d for d in dets if d["failure_class"] != "normal"]) == 1
        vibe = next(d for d in dets if d["failure_class"] == "vibration_high")
        assert vibe["is_root_cause"] is True
        assert vibe["is_downstream"] is False
        assert vibe["caused_by"] is None

    def test_power_causes_motor_failure(self):
        """Power issue at t=1 before motor failure at t=4."""
        feats = _base_features(
            bat_volt_min=8.0,
            power_onset_time=1.0,
            motor_saturation_high_count=2,
            motor_saturation_low_count=1,
            motor_failure_onset_time=4.0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        dets = dc.causal_arbiter(dets)

        pwr = next(d for d in dets if d["failure_class"] == "power_issue")
        motor = next(d for d in dets if d["failure_class"] == "motor_failure")
        assert pwr["is_root_cause"] is True
        assert motor["is_downstream"] is True
        assert motor["caused_by"] == "power_issue"

    def test_brownout_causes_ekf(self):
        feats = _base_features(
            powr_vcc_min=3.5,
            brownout_onset_time=2.0,
            ekf_sv_max=2.0,
            ekf_sv_onset_time=3.0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        dets = dc.causal_arbiter(dets)

        brownout = next(d for d in dets if d["failure_class"] == "brownout")
        ekf = next(d for d in dets if d["failure_class"] == "ekf_failure")
        assert brownout["is_root_cause"] is True
        assert ekf["is_downstream"] is True
        assert ekf["caused_by"] == "brownout"

    def test_empty_detections(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        result = dc.causal_arbiter([])
        assert result == []

    def test_normal_annotated_correctly(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        dets = dc.causal_arbiter(dets)
        normal = next(d for d in dets if d["failure_class"] == "normal")
        assert normal["is_root_cause"] is False
        assert normal["is_downstream"] is False
        assert normal["caused_by"] is None


class TestMLStub:
    def test_no_model_returns_empty(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        result = dc.classify_ml()
        assert result == []

    def test_nonexistent_path_returns_empty(self):
        feats = _base_features()
        dc = DiagnosticClassifier(feats)
        result = dc.classify_ml(model_path="/nonexistent/model.json")
        assert result == []


class TestDetectionStructure:
    def test_required_keys(self):
        feats = _base_features(vibe_x_max=50.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        for d in dets:
            assert "failure_class" in d
            assert "confidence" in d
            assert "evidence" in d
            assert "onset_time" in d

    def test_confidence_bounded(self):
        feats = _base_features(
            vibe_x_max=9999.0,
            ekf_sv_max=9999.0,
            bat_volt_min=0.0,
        )
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        for d in dets:
            assert 0.0 <= d["confidence"] <= 1.0

    def test_onset_time_is_float(self):
        feats = _base_features(vibe_x_max=50.0, vibe_onset_time=12)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        for d in dets:
            assert isinstance(d["onset_time"], float)


class TestTuningIssue:
    def test_stock_pids_plus_high_error(self):
        feats = _base_features(stock_pids_detected=True, attitude_error_max=20.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "tuning_issue" in classes

    def test_stock_pids_low_error(self):
        feats = _base_features(stock_pids_detected=True, attitude_error_max=5.0)
        dc = DiagnosticClassifier(feats)
        dets = dc.classify_rule_based()
        classes = [d["failure_class"] for d in dets]
        assert "tuning_issue" not in classes
