"""DiagnosticNarrator tests."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from flight_analyst.narrative import (
    DiagnosticNarrator,
    LLM_TOOL_DEFINITIONS,
    _EV_NAMES,
    _ERR_SUBSYS,
)


def _make_parser(
    vehicle: str = "Copter",
    firmware: str = "V4.5.1",
    duration: float = 300.0,
    modes: list[dict] | None = None,
    errors: list[dict] | None = None,
    events: list[dict] | None = None,
) -> MagicMock:
    parser = MagicMock()
    parser.get_vehicle_type.return_value = vehicle
    parser.get_firmware_version.return_value = firmware
    parser.get_flight_duration.return_value = duration
    parser.get_flight_modes.return_value = modes if modes is not None else [
        {"Mode": "Stabilize", "ModeNum": 0, "TimeUS": 1_000_000},
        {"Mode": "AltHold", "ModeNum": 2, "TimeUS": 60_000_000},
    ]
    parser.get_errors.return_value = errors if errors is not None else []
    parser.get_events.return_value = events if events is not None else [
        {"Id": 10, "TimeUS": 5_000_000},
        {"Id": 11, "TimeUS": 280_000_000},
    ]
    parser.get_params.return_value = {}
    return parser


def _make_features(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "vibe_x_mean": 8.5,
        "vibe_x_max": 15.0,
        "ekf_variance_max": 0.4,
        "gps_hdop": 1.2,
        "gps_nsats": 12,
        "batt_volt_min": 11.8,
        "compass_ofs_len": 150,
        "rcout_spread": 80,
    }
    base.update(overrides)
    return base


def _make_detections(
    failure_class: str = "high_vibration",
    confidence: int = 85,
    onset: float | None = 180.0,
    feature: str = "VIBE.VibeX",
    threshold: float | None = 30.0,
    value: float | None = 42.3,
) -> list[dict[str, Any]]:
    det: dict[str, Any] = {
        "failure_class": failure_class,
        "confidence": confidence,
    }
    if onset is not None:
        det["onset_time"] = onset
    if feature:
        det["feature"] = feature
    if threshold is not None:
        det["threshold"] = threshold
    if value is not None:
        det["value"] = value
    return [det]


def _make_audit_warnings() -> list[dict[str, Any]]:
    return [
        {
            "severity": "high",
            "param": "INS_HNTCH_ENABLE",
            "value": 0,
            "issue": "Harmonic notch filter is disabled",
            "recommendation": "Enable the harmonic notch filter",
            "wiki_url": "https://ardupilot.org/copter/docs/common-imu-notch-filtering.html",
        },
    ]


def _make_narrator(**kwargs: Any) -> DiagnosticNarrator:
    parser = kwargs.get("parser", _make_parser())
    features = kwargs.get("features", _make_features())
    detections = kwargs.get("detections", _make_detections())
    audit_warnings = kwargs.get("audit_warnings", _make_audit_warnings())
    return DiagnosticNarrator(parser, features, detections, audit_warnings)


class TestBuildTimeline:
    def test_returns_sorted_events(self):
        narrator = _make_narrator()
        timeline = narrator.build_timeline()
        times = [e["time"] for e in timeline]
        assert times == sorted(times)

    def test_includes_arm_event(self):
        narrator = _make_narrator()
        timeline = narrator.build_timeline()
        arm_events = [e for e in timeline if e["event"] == "Armed"]
        assert len(arm_events) == 1

    def test_includes_disarm_event(self):
        narrator = _make_narrator()
        timeline = narrator.build_timeline()
        disarm_events = [e for e in timeline if e["event"] == "Disarmed"]
        assert len(disarm_events) == 1

    def test_includes_mode_changes(self):
        narrator = _make_narrator()
        timeline = narrator.build_timeline()
        mode_events = [e for e in timeline if e["event"] == "Mode change"]
        assert len(mode_events) == 2

    def test_includes_anomaly_onset(self):
        narrator = _make_narrator()
        timeline = narrator.build_timeline()
        anomaly_events = [e for e in timeline if e["event"] == "Anomaly onset"]
        assert len(anomaly_events) == 1
        assert anomaly_events[0]["detail"] == "high_vibration"

    def test_includes_err_events(self):
        errors = [
            {"Subsys": 18, "ECode": 2, "TimeUS": 200_000_000},
        ]
        parser = _make_parser(errors=errors)
        narrator = _make_narrator(parser=parser)
        timeline = narrator.build_timeline()
        err_events = [e for e in timeline if e["event"].startswith("ERR:")]
        assert len(err_events) == 1
        assert "EKF_Check" in err_events[0]["event"]

    def test_empty_log_returns_empty_timeline(self):
        parser = _make_parser(modes=[], errors=[], events=[])
        narrator = _make_narrator(parser=parser, detections=[])
        timeline = narrator.build_timeline()
        assert timeline == []


class TestBuildEvidenceChain:
    def test_includes_feature_name(self):
        narrator = _make_narrator()
        evidence = narrator.build_evidence_chain(narrator.detections[0])
        assert "VIBE.VibeX" in evidence

    def test_includes_threshold(self):
        narrator = _make_narrator()
        evidence = narrator.build_evidence_chain(narrator.detections[0])
        assert "30" in evidence

    def test_includes_onset_time(self):
        narrator = _make_narrator()
        evidence = narrator.build_evidence_chain(narrator.detections[0])
        assert "T+180s" in evidence

    def test_includes_confidence(self):
        narrator = _make_narrator()
        evidence = narrator.build_evidence_chain(narrator.detections[0])
        assert "85%" in evidence

    def test_correlates_with_err_events(self):
        """When an ERR event follows shortly after onset, mention it."""
        errors = [
            {"Subsys": 19, "ECode": 2, "TimeUS": 205_000_000},
        ]
        parser = _make_parser(errors=errors)
        detections = _make_detections(onset=180.0)
        narrator = _make_narrator(parser=parser, detections=detections)
        evidence = narrator.build_evidence_chain(detections[0])
        assert "BEFORE" in evidence

    def test_minimal_detection(self):
        """Detection with only failure_class and confidence still works."""
        det = {"failure_class": "crash", "confidence": 90}
        narrator = _make_narrator(detections=[det])
        evidence = narrator.build_evidence_chain(det)
        assert "90%" in evidence


class TestBuildElimination:
    def test_produces_meaningful_text(self):
        narrator = _make_narrator()
        elim = narrator.build_elimination()
        assert len(elim) > 0
        assert "-" in elim

    def test_excludes_detected_category(self):
        """Categories matching a detection should not appear as ruled out."""
        narrator = _make_narrator(
            detections=_make_detections(failure_class="high_vibration")
        )
        elim = narrator.build_elimination()
        assert "Vibration levels within acceptable" not in elim

    def test_includes_normal_battery(self):
        narrator = _make_narrator(
            detections=_make_detections(failure_class="high_vibration")
        )
        elim = narrator.build_elimination()
        assert "Battery" in elim or "battery" in elim.lower()

    def test_includes_gps_detail(self):
        narrator = _make_narrator()
        elim = narrator.build_elimination()
        assert "HDop" in elim or "satellites" in elim


class TestBuildRecommendations:
    def test_includes_wiki_urls(self):
        narrator = _make_narrator()
        recs = narrator.build_recommendations()
        urls = [r.get("wiki_url", "") for r in recs]
        assert any("ardupilot.org" in u for u in urls)

    def test_includes_audit_warning_params(self):
        narrator = _make_narrator()
        recs = narrator.build_recommendations()
        params = [r.get("param") for r in recs if r.get("param")]
        assert "INS_HNTCH_ENABLE" in params

    def test_no_duplicate_urls_from_detections(self):
        dets = _make_detections("high_vibration") + _make_detections("high_vibration")
        narrator = _make_narrator(detections=dets)
        recs = narrator.build_recommendations()
        detection_urls = [
            r["wiki_url"] for r in recs
            if r.get("action") and "notch" not in r.get("explanation", "").lower()
        ]
        assert len(detection_urls) == len(set(detection_urls))


class TestTemplateNarrative:
    def test_has_all_sections(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "## Flight Diagnostic Report" in report
        assert "### Summary" in report
        assert "### Timeline" in report
        assert "### Analysis" in report
        assert "### Probable Cause" in report
        assert "### Recommendations" in report

    def test_summary_contains_vehicle_info(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "Copter" in report
        assert "V4.5.1" in report

    def test_summary_detects_crash(self):
        errors = [{"Subsys": 13, "ECode": 1, "TimeUS": 250_000_000}]
        parser = _make_parser(errors=errors)
        narrator = _make_narrator(parser=parser)
        report = narrator._template_narrative()
        assert "crash" in report.lower()

    def test_analysis_shows_root_cause(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "Root Cause: high_vibration" in report

    def test_analysis_shows_confidence(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "confidence: 85%" in report

    def test_probable_cause_section(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "probable cause" in report.lower()
        assert "high_vibration" in report

    def test_recommendations_include_wiki(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "ardupilot.org" in report

    def test_no_detections_produces_nominal(self):
        narrator = _make_narrator(detections=[])
        report = narrator._template_narrative()
        assert "No anomalies detected" in report

    def test_contributing_factors_from_audit(self):
        narrator = _make_narrator()
        report = narrator._template_narrative()
        assert "Contributing factor" in report
        assert "notch" in report.lower()


class TestGenerateNarrative:
    def test_without_api_key_uses_template(self):
        narrator = _make_narrator()
        report = narrator.generate_narrative()
        assert "## Flight Diagnostic Report" in report

    def test_with_api_key_calls_llm(self):
        narrator = _make_narrator()
        with patch.object(narrator, "_llm_narrative", return_value="LLM report") as mock:
            report = narrator.generate_narrative(api_key="sk-test")
        mock.assert_called_once_with("sk-test")
        assert report == "LLM report"

    def test_none_api_key_uses_template(self):
        narrator = _make_narrator()
        report = narrator.generate_narrative(api_key=None)
        assert "## Flight Diagnostic Report" in report

    def test_empty_string_api_key_uses_template(self):
        narrator = _make_narrator()
        report = narrator.generate_narrative(api_key="")
        assert "## Flight Diagnostic Report" in report


class TestLLMToolDefinitions:
    def test_all_definitions_have_type(self):
        for tool in LLM_TOOL_DEFINITIONS:
            assert tool["type"] == "function"

    def test_all_definitions_have_function_name(self):
        names = [t["function"]["name"] for t in LLM_TOOL_DEFINITIONS]
        assert "get_flight_summary" in names
        assert "get_sensor_health" in names
        assert "get_error_timeline" in names
        assert "get_parameter_audit" in names
        assert "get_wiki_recommendations" in names

    def test_all_definitions_have_valid_json_schema(self):
        for tool in LLM_TOOL_DEFINITIONS:
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_wiki_recommendations_requires_failure_class(self):
        wiki_tool = [
            t for t in LLM_TOOL_DEFINITIONS
            if t["function"]["name"] == "get_wiki_recommendations"
        ][0]
        assert "failure_class" in wiki_tool["function"]["parameters"]["required"]

    def test_definitions_are_json_serializable(self):
        """All tool definitions must be JSON-serializable for the API."""
        serialized = json.dumps(LLM_TOOL_DEFINITIONS)
        deserialized = json.loads(serialized)
        assert len(deserialized) == len(LLM_TOOL_DEFINITIONS)


def _install_fake_openai():
    import sys
    import types

    fake = types.ModuleType("openai")
    fake.OpenAI = MagicMock
    sys.modules["openai"] = fake
    return fake


class TestLLMFallback:
    def test_fallback_on_import_error(self):
        """When the openai module is missing, fall back to template."""
        import sys
        saved = sys.modules.pop("openai", None)
        try:
            narrator = _make_narrator()
            report = narrator._llm_narrative("sk-test")
            assert "template-based report" in report.lower() or "## Flight Diagnostic Report" in report
        finally:
            if saved is not None:
                sys.modules["openai"] = saved
            else:
                sys.modules.pop("openai", None)

    def test_fallback_on_api_error(self):
        fake = _install_fake_openai()
        mock_cls = MagicMock()
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.side_effect = Exception("API error")
        fake.OpenAI = mock_cls

        narrator = _make_narrator()
        report = narrator._llm_narrative("sk-test")
        assert "## Flight Diagnostic Report" in report
        assert "unavailable" in report.lower()

    def test_llm_narrative_returns_model_response(self):
        fake = _install_fake_openai()
        mock_cls = MagicMock()
        mock_client = mock_cls.return_value

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "## LLM Generated Report\nDone."
        mock_choice.message.tool_calls = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        fake.OpenAI = mock_cls

        narrator = _make_narrator()
        report = narrator._llm_narrative("sk-test")
        assert "LLM Generated Report" in report


class TestToolExecution:
    def test_execute_flight_summary(self):
        narrator = _make_narrator()
        result = narrator._execute_tool("get_flight_summary", "{}")
        assert result["vehicle_type"] == "Copter"
        assert result["firmware_version"] == "V4.5.1"
        assert result["duration_seconds"] == 300.0

    def test_execute_sensor_health(self):
        narrator = _make_narrator()
        result = narrator._execute_tool("get_sensor_health", "{}")
        assert isinstance(result, dict)

    def test_execute_error_timeline(self):
        errors = [{"Subsys": 13, "ECode": 1, "TimeUS": 250_000_000}]
        parser = _make_parser(errors=errors)
        narrator = _make_narrator(parser=parser)
        result = narrator._execute_tool("get_error_timeline", "{}")
        assert isinstance(result, list)

    def test_execute_parameter_audit(self):
        narrator = _make_narrator()
        result = narrator._execute_tool("get_parameter_audit", "{}")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_execute_wiki_recommendations(self):
        narrator = _make_narrator()
        args = json.dumps({"failure_class": "high_vibration"})
        result = narrator._execute_tool("get_wiki_recommendations", args)
        assert isinstance(result, list)

    def test_execute_unknown_tool(self):
        narrator = _make_narrator()
        result = narrator._execute_tool("nonexistent_tool", "{}")
        assert "error" in result
