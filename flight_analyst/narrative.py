"""Generates NTSB-style investigation narratives from parsed flight logs."""

import json
from typing import Any

from flight_analyst.parser import LogParser
from flight_analyst.docs import DocRecommender


# ArduPilot EV / ERR decoding tables
_EV_NAMES = {
    7: "Autopilot on",
    8: "Autopilot off",
    10: "Armed",
    11: "Disarmed",
    15: "Auto armed",
    16: "Land complete (maybe)",
    17: "Land complete",
    18: "Lost GPS",
    19: "Flip begin",
    21: "Set home",
    25: "Set simple mode on",
    26: "Set simple mode off",
    28: "Not landed",
    41: "Compass cal started",
    42: "Compass cal saved",
    43: "Compass cal failed",
    46: "EKF alt reset",
    47: "EKF yaw reset",
    49: "EKF pos reset",
    60: "RPM unhealthy",
    61: "RPM recovered",
    63: "Fence enable",
    64: "Fence disable",
    72: "Gripper grab",
    73: "Gripper release",
    90: "Surfaced",
    91: "Not surfaced",
    98: "Parachute disabled",
    99: "Parachute enabled",
    100: "Parachute released",
}

_ERR_SUBSYS = {
    1: "MainLoop",
    2: "Radio",
    3: "Compass",
    5: "RadioFailsafe",
    6: "BatteryFailsafe",
    7: "GPS",
    8: "GCS",
    9: "Fence",
    10: "FlightMode",
    12: "GPS_Glitch",
    13: "Crash",
    15: "Flip",
    17: "Parachute",
    18: "EKF_Check",
    19: "EKF_Failsafe",
    20: "BaroGlitch",
    22: "EKF_Primary",
    23: "Thrust_Loss",
    24: "Sensor_Health",
    26: "VibeFailsafe",
    27: "Internal_Error",
}

_ERR_CODES = {
    6: {  # BatteryFailsafe
        1: "Battery failsafe ON",
        0: "Battery failsafe OFF",
    },
    18: {  # EKF_Check
        2: "EKF variance bad",
        0: "EKF variance OK",
    },
    19: {  # EKF_Failsafe
        2: "EKF failsafe ON",
        0: "EKF failsafe OFF",
    },
    13: {  # Crash
        1: "Crash detected",
        0: "Crash cleared",
    },
    23: {  # Thrust_Loss
        1: "Thrust loss detected",
        0: "Thrust loss cleared",
    },
    26: {  # VibeFailsafe
        1: "Vibration failsafe ON",
        0: "Vibration failsafe OFF",
    },
}


# feature-category labels for the elimination section
_NORMAL_DESCRIPTIONS = {
    "vibration": "Vibration levels within acceptable range",
    "ekf": "EKF innovations and variances within normal limits",
    "gps": "GPS fix quality adequate throughout flight",
    "battery": "Battery voltage stable throughout",
    "compass": "Compass health normal",
    "rcout": "Motor outputs balanced",
}

# ---
# OpenAI tool definitions for LLM-enhanced mode
LLM_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_summary",
            "description": (
                "Returns vehicle type, firmware version, flight duration "
                "in seconds, and list of flight modes used."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensor_health",
            "description": (
                "Returns summary statistics for vibration, EKF, GPS, "
                "and battery sensors including min/max/mean values."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_error_timeline",
            "description": (
                "Returns decoded ERR and EV events with timestamps, "
                "subsystem names, and human-readable descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_parameter_audit",
            "description": (
                "Returns audit warnings for risky or misconfigured "
                "parameters with recommended values and wiki URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wiki_recommendations",
            "description": (
                "Returns ArduPilot wiki links relevant to a given "
                "failure class for further reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_class": {
                        "type": "string",
                        "description": (
                            "The failure class to look up, e.g. "
                            "'high_vibration', 'ekf_failure', 'crash'."
                        ),
                    },
                },
                "required": ["failure_class"],
            },
        },
    },
]

_SYSTEM_PROMPT = """\
You are an expert ArduPilot flight-log analyst producing reports in NTSB \
(National Transportation Safety Board) accident-investigation format.

You have access to tools that let you query a parsed ArduPilot dataflash log. \
Use them to gather evidence, then produce a structured diagnostic report with \
these sections:

## Flight Diagnostic Report

### Summary
Vehicle, firmware, primary mode, duration, and outcome.

### Timeline
Chronological list of significant events with T+ timestamps.

### Analysis
Root cause with confidence percentage. Evidence chain showing how sensor data \
supports the conclusion. List of ruled-out alternatives.

### Probable Cause
Single-sentence probable cause statement with contributing factors.

### Recommendations
Actionable parameter changes and wiki links.

Be precise with numbers. Cite actual sensor values and timestamps. \
When evidence is ambiguous, state the confidence level honestly.
"""


class DiagnosticNarrator:
    """Generates NTSB-format investigation narratives."""

    def __init__(
        self,
        parser: LogParser,
        features: dict[str, Any],
        detections: list[dict[str, Any]],
        audit_warnings: list[dict[str, Any]],
    ) -> None:
        self.parser = parser
        self.features = features
        self.detections = detections
        self.audit_warnings = audit_warnings
        self.doc_recommender = DocRecommender()

    def generate_narrative(self, api_key: str | None = None) -> str:
        """Generate an NTSB-style diagnostic narrative.

        Uses OpenAI function calling if api_key is given, otherwise
        falls back to a deterministic template report.
        """
        if api_key:
            return self._llm_narrative(api_key)
        return self._template_narrative()

    # --- timeline ---

    def build_timeline(self) -> list[dict[str, Any]]:
        """Reconstruct flight timeline from parser data.

        Includes arm/disarm (EV Id 10/11), mode changes, ERR events,
        and first anomaly onset from detections. Sorted by time.
        """
        events: list[dict[str, Any]] = []
        base_time = self._base_time()

        # EV messages
        for ev in self.parser.get_events():
            ev_id = int(ev.get("Id", 0))
            ev_name = _EV_NAMES.get(ev_id, f"Event {ev_id}")
            t = self._msg_time(ev, base_time)
            events.append({"time": t, "event": ev_name, "detail": f"EV Id={ev_id}"})

        # MODE messages
        for mode in self.parser.get_flight_modes():
            t = self._msg_time(mode, base_time)
            mode_name = mode.get("Mode", mode.get("ModeNum", "Unknown"))
            events.append({
                "time": t,
                "event": "Mode change",
                "detail": str(mode_name),
            })

        # ERR messages
        for err in self.parser.get_errors():
            t = self._msg_time(err, base_time)
            subsys = int(err.get("Subsys", 0))
            code = int(err.get("ECode", 0))
            subsys_name = _ERR_SUBSYS.get(subsys, f"Subsys {subsys}")
            code_desc = _ERR_CODES.get(subsys, {}).get(code, f"code {code}")
            events.append({
                "time": t,
                "event": f"ERR: {subsys_name}",
                "detail": code_desc,
            })

        # anomaly onsets from detections
        for det in self.detections:
            onset = det.get("onset_time")
            if onset is not None:
                events.append({
                    "time": float(onset),
                    "event": "Anomaly onset",
                    "detail": det.get("failure_class", "unknown"),
                })

        events.sort(key=lambda e: e["time"])
        return events

    # --- evidence chain

    def build_evidence_chain(self, detection: dict[str, Any]) -> str:
        """Build an evidence-chain string for a single detection.

        Tries to reference actual sensor values, thresholds, and timing.
        Falls back to a generic description when data is missing.
        """
        parts: list[str] = []
        fc = detection.get("failure_class", "unknown")
        confidence = detection.get("confidence", 0)
        feature = detection.get("feature", "")
        threshold = detection.get("threshold")
        value = detection.get("value")
        onset = detection.get("onset_time")

        # primary crossing statement
        if feature and value is not None and onset is not None:
            if threshold is not None:
                parts.append(
                    f"{feature} crossed {threshold} at T+{onset:.0f}s "
                    f"(measured {value:.1f})"
                )
            else:
                parts.append(
                    f"{feature} reached {value:.1f} at T+{onset:.0f}s"
                )
        elif feature and value is not None:
            parts.append(f"{feature} measured at {value:.1f}")

        # correlate with ERR events
        err_events = [
            e for e in self.build_timeline()
            if e["event"].startswith("ERR:")
        ]
        if onset is not None and err_events:
            for err_ev in err_events:
                delta = err_ev["time"] - onset
                if 0 < delta < 120:
                    parts.append(
                        f"{abs(delta):.0f}s BEFORE {err_ev['event']} "
                        f"({err_ev['detail']}) at T+{err_ev['time']:.0f}s"
                    )
                    break

        parts.append(f"Confidence: {confidence}%")

        if not parts:
            parts.append(f"{fc} detected with {confidence}% confidence")

        return " -- ".join(parts)

    # --- elimination

    def build_elimination(self) -> str:
        """Build a 'ruled out' block for normal feature categories."""
        lines: list[str] = []
        detection_classes = {
            d.get("failure_class", "").lower() for d in self.detections
        }

        for category, description in _NORMAL_DESCRIPTIONS.items():
            if any(category in dc for dc in detection_classes):
                continue

            detail = self._elimination_detail(category)
            if detail:
                lines.append(f"{description}: {detail}")
            else:
                lines.append(description)

        return "\n".join(f"- {line}" for line in lines) if lines else "- No categories ruled out"

    def _elimination_detail(self, category: str) -> str:
        """Short supporting detail for a normal category."""
        feat = self.features

        if category == "vibration":
            vx = feat.get("vibe_x_mean") or feat.get("vibration_x_mean")
            if vx is not None:
                return f"mean vibration {vx:.1f} m/s^2"

        if category == "battery":
            batt = feat.get("batt_volt_min") or feat.get("battery_voltage_min")
            if batt is not None:
                return f"min voltage {batt:.1f}V"

        if category == "gps":
            hdop = feat.get("gps_hdop") or feat.get("gps_hdop_mean")
            nsats = feat.get("gps_nsats") or feat.get("gps_nsats_mean")
            parts = []
            if hdop is not None:
                parts.append(f"HDop {hdop:.1f}")
            if nsats is not None:
                parts.append(f"{nsats:.0f} satellites")
            if parts:
                return " with ".join(parts)

        if category == "ekf":
            var = feat.get("ekf_variance_max") or feat.get("ekf_var_max")
            if var is not None:
                return f"peak variance {var:.2f}"

        if category == "compass":
            ofs = feat.get("compass_ofs_len") or feat.get("compass_offset_length")
            if ofs is not None:
                return f"offset length {ofs:.0f}"

        if category == "rcout":
            spread = feat.get("rcout_spread") or feat.get("motor_spread")
            if spread is not None:
                return f"max spread {spread:.0f} us"

        return ""

    # --- recommendations

    def build_recommendations(self) -> list[dict[str, Any]]:
        """Combine detection-specific docs with audit warnings."""
        recs: list[dict[str, Any]] = []
        vehicle = self.parser.get_vehicle_type() or "Copter"

        # from DocRecommender
        seen_urls: set[str] = set()
        for det in self.detections:
            fc = det.get("failure_class", "")
            docs = self.doc_recommender.recommend(fc, vehicle_type=vehicle)
            for doc in docs:
                url = doc["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    recs.append({
                        "action": doc["title"],
                        "explanation": doc["relevance_note"],
                        "wiki_url": url,
                    })

        # audit warnings
        for warn in self.audit_warnings:
            url = warn.get("wiki_url", "")
            rec: dict[str, Any] = {
                "explanation": warn.get("recommendation", warn.get("issue", "")),
                "wiki_url": url,
            }
            if warn.get("param"):
                rec["param"] = warn["param"]
                rec["value"] = warn.get("value")
            else:
                rec["action"] = warn.get("issue", "Review configuration")
            recs.append(rec)

        return recs

    # --- template narrative

    def _template_narrative(self) -> str:
        sections: list[str] = []

        sections.append("## Flight Diagnostic Report")
        sections.append(self._build_summary_section())
        sections.append(self._build_timeline_section())
        sections.append(self._build_analysis_section())
        sections.append(self._build_probable_cause_section())
        sections.append(self._build_recommendations_section())

        return "\n\n".join(sections)

    def _build_summary_section(self) -> str:
        vehicle = self.parser.get_vehicle_type() or "Unknown vehicle"
        firmware = self.parser.get_firmware_version() or "unknown firmware"
        duration = self.parser.get_flight_duration()
        modes = self.parser.get_flight_modes()
        primary_mode = modes[0].get("Mode", "Unknown") if modes else "Unknown"

        errors = self.parser.get_errors()
        crash = any(
            int(e.get("Subsys", 0)) == 13 and int(e.get("ECode", 0)) == 1
            for e in errors
        )
        if crash:
            outcome = "Flight ended in a crash."
        elif errors:
            outcome = "Flight completed with errors."
        else:
            outcome = "Flight completed nominally."

        return (
            f"### Summary\n"
            f"{vehicle} {firmware}, {primary_mode} mode, "
            f"{duration:.0f}s flight.\n"
            f"{outcome}"
        )

    def _build_timeline_section(self) -> str:
        timeline = self.build_timeline()
        if not timeline:
            return "### Timeline\nNo events recorded."
        lines = ["### Timeline"]
        for ev in timeline:
            lines.append(
                f"- T+{ev['time']:.0f}s: {ev['event']} — {ev['detail']}"
            )
        return "\n".join(lines)

    def _build_analysis_section(self) -> str:
        lines = ["### Analysis"]

        if self.detections:
            primary = self.detections[0]
            fc = primary.get("failure_class", "Unknown")
            conf = primary.get("confidence", 0)
            lines.append(f"**Root Cause: {fc}** (confidence: {conf}%)")
            lines.append("")
            lines.append("Evidence chain:")

            for i, det in enumerate(self.detections, 1):
                evidence = self.build_evidence_chain(det)
                lines.append(f"{i}. {evidence}")
        else:
            lines.append("**No anomalies detected.**")

        lines.append("")
        lines.append("**Ruled out:**")
        lines.append(self.build_elimination())

        return "\n".join(lines)

    def _build_probable_cause_section(self) -> str:
        lines = ["### Probable Cause"]
        if self.detections:
            primary = self.detections[0]
            fc = primary.get("failure_class", "Unknown")
            conf = primary.get("confidence", 0)
            lines.append(
                f"The probable cause of this incident is {fc} "
                f"(confidence: {conf}%)."
            )
            if self.audit_warnings:
                factors = [
                    w.get("issue", w.get("param", "unknown"))
                    for w in self.audit_warnings
                ]
                lines.append(
                    f"Contributing factor(s): {'; '.join(factors)}"
                )
        else:
            lines.append("No probable cause identified -- flight nominal.")

        return "\n".join(lines)

    def _build_recommendations_section(self) -> str:
        recs = self.build_recommendations()
        if not recs:
            return "### Recommendations\nNo specific recommendations."
        lines = ["### Recommendations"]
        for i, rec in enumerate(recs, 1):
            if "param" in rec:
                lines.append(
                    f"{i}. **{rec['param']},{rec.get('value', '?')}** "
                    f"-- {rec['explanation']}"
                )
            else:
                lines.append(
                    f"{i}. {rec.get('action', 'Review')} -- "
                    f"{rec['explanation']}"
                )
            url = rec.get("wiki_url", "")
            if url:
                lines.append(f"   See: {url}")
        return "\n".join(lines)

    # --- LLM-enhanced narrative

    def _llm_narrative(self, api_key: str) -> str:
        """Use OpenAI function calling for a richer narrative.

        Falls back to template on any API error.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Diagnose this ArduPilot flight log. Use the "
                        "available tools to gather data, then produce a "
                        "structured NTSB-format diagnostic report."
                    ),
                },
            ]

            # FIXME: should probably back off or retry on rate limits
            max_iterations = 5
            for _ in range(max_iterations):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=LLM_TOOL_DEFINITIONS,
                    tool_choice="auto",
                )

                choice = response.choices[0]

                if choice.finish_reason == "stop":
                    return choice.message.content or self._template_narrative()

                if choice.message.tool_calls:
                    messages.append(choice.message)
                    for tool_call in choice.message.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = tool_call.function.arguments
                        result = self._execute_tool(fn_name, fn_args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        })
                else:
                    return (
                        choice.message.content or self._template_narrative()
                    )

            return self._template_narrative()

        except Exception:
            template = self._template_narrative()
            return (
                f"*Note: LLM-enhanced analysis unavailable, "
                f"using template-based report.*\n\n{template}"
            )

    def _execute_tool(self, name: str, args_json: str) -> Any:
        """Dispatch a tool call from the LLM."""
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError:
            args = {}

        if name == "get_flight_summary":
            return self._tool_flight_summary()
        elif name == "get_sensor_health":
            return self._tool_sensor_health()
        elif name == "get_error_timeline":
            return self._tool_error_timeline()
        elif name == "get_parameter_audit":
            return self._tool_parameter_audit()
        elif name == "get_wiki_recommendations":
            fc = args.get("failure_class", "")
            return self._tool_wiki_recommendations(fc)
        else:
            return {"error": f"Unknown tool: {name}"}

    # tool implementations

    def _tool_flight_summary(self) -> dict[str, Any]:
        vehicle = self.parser.get_vehicle_type() or "Unknown"
        firmware = self.parser.get_firmware_version() or "Unknown"
        duration = self.parser.get_flight_duration()
        modes = self.parser.get_flight_modes()
        mode_list = [m.get("Mode", str(m.get("ModeNum", "?"))) for m in modes]
        return {
            "vehicle_type": vehicle,
            "firmware_version": firmware,
            "duration_seconds": round(duration, 1),
            "modes": mode_list,
        }

    def _tool_sensor_health(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key_prefix in ("vibe_x", "vibration_x"):
            for stat in ("mean", "max"):
                k = f"{key_prefix}_{stat}"
                if k in self.features:
                    result.setdefault("vibration", {})[f"x_{stat}"] = self.features[k]

        for k in ("ekf_variance_max", "ekf_var_max", "ekf_variance_mean"):
            if k in self.features:
                result.setdefault("ekf", {})[k] = self.features[k]

        for k in ("gps_hdop", "gps_hdop_mean", "gps_nsats", "gps_nsats_mean"):
            if k in self.features:
                result.setdefault("gps", {})[k] = self.features[k]

        for k in ("batt_volt_min", "battery_voltage_min", "batt_volt_mean"):
            if k in self.features:
                result.setdefault("battery", {})[k] = self.features[k]

        return result

    def _tool_error_timeline(self) -> list[dict[str, Any]]:
        timeline = self.build_timeline()
        return [
            e for e in timeline
            if e["event"].startswith("ERR:") or e["event"] in _EV_NAMES.values()
        ]

    def _tool_parameter_audit(self) -> list[dict[str, Any]]:
        return self.audit_warnings

    def _tool_wiki_recommendations(self, failure_class: str) -> list[dict[str, str]]:
        vehicle = self.parser.get_vehicle_type() or "Copter"
        return self.doc_recommender.recommend(failure_class, vehicle_type=vehicle)

    # helpers

    def _base_time(self) -> float:
        """Earliest timestamp across all cached messages."""
        all_msgs = (
            self.parser.get_events()
            + self.parser.get_flight_modes()
            + self.parser.get_errors()
        )
        times = []
        for m in all_msgs:
            t = self._raw_time(m)
            if t is not None:
                times.append(t)
        return min(times) if times else 0.0

    @staticmethod
    def _raw_time(msg: dict[str, Any]) -> float | None:
        if "TimeUS" in msg:
            return float(msg["TimeUS"]) * 1e-6
        if "_timestamp" in msg:
            return float(msg["_timestamp"])
        return None

    def _msg_time(self, msg: dict[str, Any], base_time: float) -> float:
        """Get a message's time relative to base_time."""
        raw = self._raw_time(msg)
        if raw is not None:
            return raw - base_time
        return 0.0
