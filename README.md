# ArduPilot Flight Analyst

Flight log diagnosis tool for ArduPilot -- anomaly classification, SITL fault injection, and diagnostic report generation.

## Features

- Auto-detects EKF3 (XKF4) vs EKF2 (NKF4/EKF4), decodes innovation variances and fault/timeout bitmasks
- Processes 20+ DataFlash message types: VIBE, ATT, GPS, BAT, RCOU, POWR, PM, CTUN, MAG, RCIN, ERR, EV, MODE, etc.
- Rule-based classifier covering vibration, EKF failure, compass interference, GPS glitch, motor issues, power problems, RC failsafe, brownout, and more
- Causal arbiter that uses onset times to separate root causes from downstream symptoms
- NTSB-format reports with timeline, evidence chain, elimination reasoning, and parameter recommendations
- Pre-flight parameter audit -- flags stock PIDs, missing notch filter, disabled failsafes
- Maps diagnoses to relevant ArduPilot wiki pages
- SITL fault injection with 10 scenarios for generating labeled training data
- Optional OpenAI integration for LLM-generated narratives

## Quick Start

```bash
git clone https://github.com/your-username/ardupilot-flight-analyst.git
cd ardupilot-flight-analyst
pip install -e .

flight-analyst path/to/logfile.bin --health
```

### Requirements

- Python 3.10+
- pymavlink >= 2.4.41
- numpy >= 1.24.0
- scikit-learn >= 1.3.0 (for future ML classifiers)
- xgboost >= 2.0.0 (for future ML classifiers)
- openai >= 1.0.0 (optional, for LLM narratives)

## CLI Usage

The `flight-analyst` command provides five analysis modes:

### Health summary

```bash
flight-analyst logfile.bin --health
```

Prints vibration levels, EKF variance peaks, battery stats, GPS quality, and error/event counts. Default mode when no flag is specified.

```
----------------------------------------------
  Log Summary
----------------------------------------------
  Vehicle type:          Copter
  Firmware:              V4.5.7
  EKF version:           EKF3
  Duration:              312.4s
  Message types:         24

----------------------------------------------
  Health Summary
----------------------------------------------
  Vibration
    X max:               12.3
    Z max:               18.7
    Clip count:          0
  EKF
    Vel variance max:    0.42
    Fault count:         0
  ...
```

### Parameter audit

```bash
flight-analyst logfile.bin --audit
```

Checks parameters against known-bad configurations: stock default PIDs, disabled failsafes, missing notch filter, disabled compass, and more. Each warning includes severity, a fix recommendation, and a link to the relevant wiki page.

### Anomaly diagnosis

```bash
flight-analyst logfile.bin --diagnose
```

Runs the rule-based classifier, then applies the causal arbiter to separate root causes from downstream effects. Output includes confidence scores, evidence strings, onset times, and documentation links.

### Investigation narrative

```bash
flight-analyst logfile.bin --narrative
flight-analyst logfile.bin --narrative --openai-key sk-...
```

Generates an NTSB-style investigation report:

- Synopsis and flight metadata
- Chronological timeline of anomalies
- Evidence chain and elimination reasoning
- Probable cause determination
- Parameter recommendations with wiki links

With an OpenAI API key, the narrative is LLM-generated using extracted features as context. Without one, a structured template-based report is produced.

### JSON output

```bash
flight-analyst logfile.bin --json
```

Outputs the complete analysis as a single JSON document for piping into other tools.

## Library Usage

```python
from flight_analyst.parser import LogParser
from flight_analyst.features import FeatureExtractor
from flight_analyst.audit import audit as run_audit
from flight_analyst.classifier import DiagnosticClassifier
from flight_analyst.docs import DocRecommender

# Parse and extract
parser = LogParser("logfile.bin")
features = FeatureExtractor(parser).extract_all()

# Audit parameters
warnings = run_audit(parser.get_params())

# Classify and determine causality
classifier = DiagnosticClassifier(features)
detections = classifier.classify_rule_based()
detections = classifier.causal_arbiter(detections)

# Get documentation links
recommender = DocRecommender()
for d in detections:
    docs = recommender.recommend(d["failure_class"], "Copter")
```

See `examples/analyze_log.py` for a complete working example.

## Architecture

```
                           ArduPilot .bin log
                                  |
                                  v
                      +-----------------------+
                      |      LogParser        |
                      |  (pymavlink wrapper)  |
                      +-----------+-----------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
          +-----------------+         +-----------------+
          | FeatureExtractor|         | ParameterAudit  |
          |  (20+ msg types)|         | (8 check rules) |
          +--------+--------+         +--------+--------+
                   |                           |
                   v                           |
          +-----------------+                  |
          | Diagnostic      |                  |
          | Classifier      |                  |
          +--------+--------+                  |
                   |                           |
                   v                           |
          +-----------------+                  |
          | Causal Arbiter  |                  |
          | (temporal order)|                  |
          +--------+--------+                  |
                   |                           |
                   +-------------+-------------+
                                 |
                                 v
                    +------------------------+
                    | Narrative / JSON output |
                    | + DocRecommender       |
                    +------------------------+
```

## SITL Fault Injection

The `flight_analyst.sitl.inject` module provides 10 fault-injection scenarios for ArduPilot SITL. Each scenario injects a specific fault (motor failure, GPS glitch, compass interference, etc.) and produces a `.bin` log with a known ground-truth label.

```bash
python -m flight_analyst.sitl.inject --scenario vibration_high
```

Useful for validating classifier accuracy and generating training data.

## Project Structure

```
ardupilot-flight-analyst/
  flight_analyst/
    __init__.py
    parser.py            # LogParser (pymavlink wrapper)
    features.py          # FeatureExtractor
    classifier.py        # DiagnosticClassifier + causal arbiter
    audit.py             # Parameter audit checks
    docs.py              # DocRecommender (wiki URL mapping)
    cli.py               # CLI entry point
    sitl/
      __init__.py
      inject.py          # SITL fault injection
  examples/
    analyze_log.py
  tests/
  pyproject.toml
  requirements.txt
```

## Links

- [ArduPilot](https://ardupilot.org)
- [pymavlink](https://github.com/ArduPilot/pymavlink)
- [ArduPilot Log Messages Reference](https://ardupilot.org/copter/docs/logmessages.html)
- [ArduPilot Troubleshooting](https://ardupilot.org/copter/docs/troubleshooting.html)

Prototype for the [AI-Assisted Log Diagnosis](https://ardupilot.org/dev/docs/gsoc-ideas-list.html) GSoC 2026 project.
