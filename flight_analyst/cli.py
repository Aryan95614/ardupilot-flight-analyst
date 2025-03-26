#!/usr/bin/env python3
"""CLI entry point for ArduPilot Flight Analyst."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from flight_analyst import __version__
from flight_analyst.parser import LogParser
from flight_analyst.features import FeatureExtractor
from flight_analyst.audit import audit as run_audit
from flight_analyst.classifier import DiagnosticClassifier
from flight_analyst.docs import DocRecommender

_SEP = "-" * 72


def _header(title: str) -> str:
    return f"\n{_SEP}\n  {title}\n{_SEP}"


def _kv(key: str, value: Any, width: int = 22) -> str:
    return f"  {key:<{width}} {value}"


def _print_log_info(parser: LogParser) -> None:
    vehicle = parser.get_vehicle_type() or "Unknown"
    firmware = parser.get_firmware_version() or "Unknown"
    ekf = parser.get_ekf_version()
    ekf_str = f"EKF{ekf}" if ekf else "Unknown"
    duration = parser.get_flight_duration()
    msg_types = parser.get_message_types()

    print(_header("Log Summary"))
    print(_kv("Vehicle type:", vehicle))
    print(_kv("Firmware:", firmware))
    print(_kv("EKF version:", ekf_str))
    print(_kv("Duration:", f"{duration:.1f}s"))
    print(_kv("Message types:", len(msg_types)))


def _print_health(features: dict[str, Any], parser: LogParser) -> None:
    print(_header("Health Summary"))

    print("\n  Vibration")
    print(_kv("  X max:", f"{features.get('vibe_x_max', 'n/a')}"))
    print(_kv("  Y max:", f"{features.get('vibe_y_max', 'n/a')}"))
    print(_kv("  Z max:", f"{features.get('vibe_z_max', 'n/a')}"))
    print(_kv("  Clip count:", f"{features.get('clips', 'n/a')}"))

    print("\n  EKF")
    print(_kv("  Vel variance max:", f"{features.get('ekf_sv_max', 'n/a')}"))
    print(_kv("  Pos variance max:", f"{features.get('ekf_sp_max', 'n/a')}"))
    print(_kv("  Fault count:", f"{features.get('ekf_fault_count', 'n/a')}"))
    print(_kv("  Timeout count:", f"{features.get('ekf_timeout_count', 'n/a')}"))

    print("\n  Battery")
    print(_kv("  Voltage min:", f"{features.get('bat_volt_min', 'n/a')}"))
    print(_kv("  Current max:", f"{features.get('bat_curr_max', 'n/a')}"))
    print(_kv("  mAh consumed:", f"{features.get('bat_mah_consumed', 'n/a')}"))

    print("\n  GPS")
    print(_kv("  Fix type mode:", f"{features.get('gps_fix_mode', 'n/a')}"))
    print(_kv("  Satellites min:", f"{features.get('gps_nsat_min', 'n/a')}"))
    print(_kv("  HDop max:", f"{features.get('gps_hdop_max', 'n/a')}"))

    errors = parser.get_errors()
    print("\n  Errors/Events")
    print(_kv("  ERR messages:", len(errors)))
    print(_kv("  EV messages:", len(parser.get_events())))
    if errors:
        from flight_analyst.features import ERR_SUBSYS
        subsys_counts: dict[str, int] = {}
        for e in errors:
            subsys_id = e.get("Subsys", 0)
            name = ERR_SUBSYS.get(subsys_id, f"Subsys-{subsys_id}")
            subsys_counts[name] = subsys_counts.get(name, 0) + 1
        for name, count in sorted(subsys_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")


def _print_audit(warnings: list[dict[str, Any]]) -> None:
    print(_header("Parameter Audit"))

    if not warnings:
        print("  No parameter warnings found. Configuration looks good.")
        return

    print(f"  {len(warnings)} warning(s) found:\n")
    for i, w in enumerate(warnings, 1):
        severity = w["severity"].upper()
        print(f"  {i}. [{severity}] {w['issue']}")
        print(f"     Parameter: {w['param']} = {w['value']}")
        print(f"     Fix:       {w['recommendation']}")
        print(f"     Docs:      {w['wiki_url']}")
        print()


def _print_diagnose(
    detections: list[dict[str, Any]], vehicle_type: str
) -> None:
    print(_header("Diagnostic Classification"))

    recommender = DocRecommender()

    root_causes = [d for d in detections if d.get("is_root_cause")]
    downstream = [d for d in detections if d.get("is_downstream")]
    other = [
        d
        for d in detections
        if not d.get("is_root_cause") and not d.get("is_downstream")
        and d["failure_class"] != "normal"
    ]

    if not root_causes and not other:
        print("  No anomalies detected. Flight appears normal.")
        return

    if root_causes:
        print("\n  ROOT CAUSE(S):")
        for d in root_causes:
            _print_detection(d, recommender, vehicle_type, indent=4)

    if downstream:
        print("\n  DOWNSTREAM EFFECTS:")
        for d in downstream:
            _print_detection(d, recommender, vehicle_type, indent=4)

    if other:
        print("\n  OTHER DETECTIONS:")
        for d in other:
            _print_detection(d, recommender, vehicle_type, indent=4)


def _print_detection(
    d: dict[str, Any],
    recommender: DocRecommender,
    vehicle_type: str,
    indent: int = 4,
) -> None:
    pad = " " * indent
    print(f"{pad}{d['failure_class']}  (confidence: {d['confidence']:.0%})")
    print(f"{pad}  Evidence: {d.get('evidence', 'n/a')}")
    if d.get("onset_time"):
        print(f"{pad}  Onset:    {d['onset_time']:.1f}s")
    if d.get("caused_by"):
        print(f"{pad}  Cause:    {d['caused_by']}")

    docs = recommender.recommend(d["failure_class"], vehicle_type or "Copter")
    if docs:
        print(f"{pad}  Docs:")
        for doc in docs:
            print(f"{pad}    - {doc['title']}: {doc['url']}")
    print()


def _print_narrative(
    parser: LogParser,
    features: dict[str, Any],
    detections: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    openai_key: str | None = None,
) -> None:
    print(_header("Investigation Narrative"))

    vehicle = parser.get_vehicle_type() or "Unknown"
    firmware = parser.get_firmware_version() or "Unknown"
    ekf = parser.get_ekf_version()
    duration = parser.get_flight_duration()

    root_causes = [d for d in detections if d.get("is_root_cause")]
    downstream = [d for d in detections if d.get("is_downstream")]
    recommender = DocRecommender()

    if openai_key:
        narrative = _generate_llm_narrative(
            openai_key, parser, features, detections, warnings
        )
        if narrative:
            print(narrative)
            return

    print(f"""
  FLIGHT INVESTIGATION REPORT
  {"=" * 40}

  Aircraft:     {vehicle}
  Firmware:     {firmware}
  EKF:          EKF{ekf or '?'}
  Duration:     {duration:.1f}s
  Warnings:     {len(warnings)} parameter issue(s)
""")

    print("  TIMELINE OF ANOMALIES")
    print("  " + "-" * 40)
    sorted_dets = sorted(
        [d for d in detections if d["failure_class"] != "normal"],
        key=lambda d: d.get("onset_time", float("inf")),
    )
    if not sorted_dets:
        print("  No anomalies detected during this flight.")
    else:
        for d in sorted_dets:
            onset = d.get("onset_time", 0)
            role = ""
            if d.get("is_root_cause"):
                role = " [ROOT CAUSE]"
            elif d.get("is_downstream"):
                role = " [DOWNSTREAM]"
            print(
                f"  T+{onset:>7.1f}s  {d['failure_class']}"
                f"  ({d['confidence']:.0%}){role}"
            )
            print(f"              Evidence: {d.get('evidence', 'n/a')}")

    if root_causes:
        print(f"\n  PROBABLE CAUSE")
        print("  " + "-" * 40)
        for rc in root_causes:
            print(f"  {rc['failure_class']} (confidence {rc['confidence']:.0%})")
            print(f"  Evidence: {rc.get('evidence', 'n/a')}")

    if downstream:
        print(f"\n  CAUSAL REASONING")
        print("  " + "-" * 40)
        for ds in downstream:
            cause = ds.get("caused_by", "unknown")
            print(
                f"  {ds['failure_class']} classified as downstream effect "
                f"of {cause}"
            )

    print(f"\n  RECOMMENDATIONS")
    print("  " + "-" * 40)
    rec_set: set[str] = set()
    for d in sorted_dets:
        docs = recommender.recommend(
            d["failure_class"], vehicle or "Copter"
        )
        for doc in docs:
            line = f"  - {doc['title']}: {doc['url']}"
            if line not in rec_set:
                rec_set.add(line)
                print(line)

    if warnings:
        print(f"\n  PARAMETER FIXES")
        print("  " + "-" * 40)
        for w in warnings:
            print(f"  - [{w['severity'].upper()}] {w['recommendation']}")
            print(f"    ({w['param']} = {w['value']})")

    print()


def _generate_llm_narrative(
    api_key: str,
    parser: LogParser,
    features: dict[str, Any],
    detections: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> str | None:
    try:
        from openai import OpenAI
    except ImportError:
        print("  [warning] openai package not installed; falling back to rule-based narrative.")
        return None

    vehicle = parser.get_vehicle_type() or "Unknown"
    firmware = parser.get_firmware_version() or "Unknown"
    duration = parser.get_flight_duration()

    det_summary = json.dumps(detections, indent=2, default=str)
    warn_summary = json.dumps(warnings, indent=2, default=str)

    feat_subset = {
        k: v
        for k, v in features.items()
        if not k.startswith("_error_") and isinstance(v, (int, float, str, bool))
    }
    feat_summary = json.dumps(feat_subset, indent=2, default=str)

    prompt = f"""You are an expert ArduPilot flight analyst. Write an NTSB-style
investigation narrative for the following flight log analysis.

Vehicle: {vehicle}, Firmware: {firmware}, Duration: {duration:.1f}s

Features:
{feat_summary}

Detections:
{det_summary}

Parameter warnings:
{warn_summary}

Structure the report with these sections:
1. Synopsis
2. Timeline of anomalies
3. Evidence chain
4. Elimination reasoning (why downstream effects are not the root cause)
5. Probable cause
6. Recommendations with specific ArduPilot parameter changes

Be concise, technical, and reference specific parameter names and values."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"  [warning] OpenAI API call failed: {exc}")
        print("  Falling back to rule-based narrative.\n")
        return None


def _build_json_output(
    parser: LogParser,
    features: dict[str, Any],
    detections: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "log_info": {
            "vehicle_type": parser.get_vehicle_type(),
            "firmware": parser.get_firmware_version(),
            "ekf_version": parser.get_ekf_version(),
            "duration_s": round(parser.get_flight_duration(), 2),
            "message_types": parser.get_message_types(),
        },
        "features": {
            k: v
            for k, v in features.items()
            if not k.startswith("_error_")
        },
        "detections": detections,
        "parameter_warnings": warnings,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flight-analyst",
        description="ArduPilot Flight Analyst -- flight log diagnosis tool.",
    )
    p.add_argument(
        "logfile",
        help="Path to an ArduPilot .bin or .log file",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--health",
        action="store_true",
        help="Print a health summary (vibration, EKF, battery, GPS, errors)",
    )
    mode.add_argument(
        "--audit",
        action="store_true",
        help="Run a parameter audit and print warnings",
    )
    mode.add_argument(
        "--diagnose",
        action="store_true",
        help="Classify anomalies with causal analysis",
    )
    mode.add_argument(
        "--narrative",
        action="store_true",
        help="Generate a full investigation narrative",
    )
    mode.add_argument(
        "--json",
        action="store_true",
        help="Output the full analysis as JSON",
    )

    p.add_argument(
        "--openai-key",
        metavar="KEY",
        default=None,
        help="OpenAI API key for LLM-enhanced narratives (optional)",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logpath = Path(args.logfile)
    if not logpath.exists():
        print(f"Error: file not found: {logpath}", file=sys.stderr)
        sys.exit(1)
    if not logpath.is_file():
        print(f"Error: not a file: {logpath}", file=sys.stderr)
        sys.exit(1)

    try:
        log = LogParser(str(logpath))
    except Exception as exc:
        print(f"Error: failed to parse log: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        _print_log_info(log)

    # default to --health when nothing specified
    if not any([args.health, args.audit, args.diagnose, args.narrative, args.json]):
        args.health = True

    features: dict[str, Any] = {}
    if args.health or args.diagnose or args.narrative or args.json:
        try:
            extractor = FeatureExtractor(log)
            features = extractor.extract_all()
        except Exception as exc:
            print(f"Error: feature extraction failed: {exc}", file=sys.stderr)
            sys.exit(1)

    warnings: list[dict[str, Any]] = []
    if args.audit or args.narrative or args.json:
        try:
            params = log.get_params()
            warnings = run_audit(params)
        except Exception as exc:
            print(f"Error: parameter audit failed: {exc}", file=sys.stderr)
            sys.exit(1)

    detections: list[dict[str, Any]] = []
    if args.diagnose or args.narrative or args.json:
        try:
            classifier = DiagnosticClassifier(features)
            detections = classifier.classify_rule_based()
            detections = classifier.causal_arbiter(detections)
        except Exception as exc:
            print(f"Error: classification failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.health:
        _print_health(features, log)
    elif args.audit:
        _print_audit(warnings)
    elif args.diagnose:
        _print_diagnose(detections, log.get_vehicle_type())
    elif args.narrative:
        _print_narrative(log, features, detections, warnings, args.openai_key)
    elif args.json:
        if not warnings and not args.audit:
            params = log.get_params()
            warnings = run_audit(params)
        output = _build_json_output(log, features, detections, warnings)
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
