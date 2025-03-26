#!/usr/bin/env python3
"""Example: analyze an ArduPilot .bin log file."""

import sys

from flight_analyst.parser import LogParser
from flight_analyst.features import FeatureExtractor
from flight_analyst.audit import audit as run_audit
from flight_analyst.classifier import DiagnosticClassifier
from flight_analyst.docs import DocRecommender


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_log.py <logfile.bin>")
        sys.exit(1)

    log_path = sys.argv[1]

    parser = LogParser(log_path)
    print(f"Vehicle: {parser.get_vehicle_type()}")
    print(f"Firmware: {parser.get_firmware_version()}")
    print(f"EKF version: EKF{parser.get_ekf_version()}")
    print(f"Duration: {parser.get_flight_duration():.0f}s")

    extractor = FeatureExtractor(parser)
    features = extractor.extract_all()

    params = parser.get_params()
    warnings = run_audit(params)

    if warnings:
        print(f"\n{len(warnings)} parameter warning(s) found:")
        for w in warnings:
            print(f"  [{w['severity'].upper()}] {w['issue']}")
            print(f"    -> {w['recommendation']}")
    else:
        print("\nNo parameter warnings.")

    classifier = DiagnosticClassifier(features)
    detections = classifier.classify_rule_based()
    detections = classifier.causal_arbiter(detections)

    print(f"\n{len(detections)} detection(s):")
    recommender = DocRecommender()
    vehicle = parser.get_vehicle_type() or "Copter"

    for d in detections:
        role = ""
        if d.get("is_root_cause"):
            role = " [ROOT CAUSE]"
        elif d.get("is_downstream"):
            role = " [DOWNSTREAM]"
        print(f"  {d['failure_class']} ({d['confidence']:.0%}){role}")
        print(f"    Evidence: {d.get('evidence', 'n/a')}")

        docs = recommender.recommend(d["failure_class"], vehicle)
        for doc in docs:
            print(f"    -> {doc['title']}: {doc['url']}")


if __name__ == "__main__":
    main()
