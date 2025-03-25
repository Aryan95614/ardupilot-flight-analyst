"""Rule-based and ML classification for flight anomalies."""

import os
from typing import Any

# Canonical failure classes
FAILURE_CLASSES = [
    "vibration_high",
    "ekf_failure",
    "compass_interference",
    "gps_glitch",
    "motor_imbalance",
    "motor_failure",
    "power_issue",
    "rc_failsafe",
    "thrust_loss",
    "tuning_issue",
    "mechanical_failure",
    "esc_desync",
    "brownout",
    "scheduling_overrun",
    "normal",
]

# Known causal chains: (cause, effect)
# cause can produce effect as a downstream symptom
CAUSAL_CHAINS = [
    ("vibration_high", "ekf_failure"),
    ("motor_failure", "vibration_high"),
    ("motor_failure", "compass_interference"),
    ("power_issue", "motor_failure"),
    ("power_issue", "vibration_high"),
    ("brownout", "ekf_failure"),
    ("scheduling_overrun", "ekf_failure"),
]


def _safe_get(features: dict, key: str, default: Any = 0.0) -> Any:
    return features.get(key, default)


def _confidence_above(value: float, threshold: float, scale: float | None = None) -> float:
    """Confidence in [0,1] based on how far value exceeds threshold.

    scale controls the denominator -- defaults to threshold so that
    double the threshold gives confidence 1.0.
    """
    if value <= threshold:
        return 0.0
    if scale is None:
        scale = threshold if threshold != 0 else 1.0
    return min(1.0, (value - threshold) / scale)


def _confidence_below(value: float, threshold: float, scale: float | None = None) -> float:
    """Confidence for metrics that are bad when below a threshold."""
    if value >= threshold:
        return 0.0
    if scale is None:
        scale = threshold if threshold != 0 else 1.0
    return min(1.0, (threshold - value) / scale)


class DiagnosticClassifier:
    """Classify flight anomalies from a pre-computed feature dict."""

    def __init__(self, features: dict) -> None:
        self.features = features

    # --- rule-based detection helpers (one per failure class) ---

    def _detect_vibration_high(self) -> dict | None:
        vibe_x = _safe_get(self.features, "vibe_x_max", 0.0)
        vibe_z = _safe_get(self.features, "vibe_z_max", 0.0)
        clips = _safe_get(self.features, "clips", 0)
        threshold = 30.0

        if vibe_x > threshold or vibe_z > threshold or clips > 0:
            peak = max(vibe_x, vibe_z)
            conf = _confidence_above(peak, threshold, threshold)
            if clips > 0 and conf < 0.5:
                conf = max(conf, 0.5)
            onset = _safe_get(self.features, "vibe_onset_time", 0.0)
            evidence_parts = []
            if vibe_x > threshold:
                evidence_parts.append(f"vibe_x_max={vibe_x:.1f}")
            if vibe_z > threshold:
                evidence_parts.append(f"vibe_z_max={vibe_z:.1f}")
            if clips > 0:
                evidence_parts.append(f"clips={clips}")
            return {
                "failure_class": "vibration_high",
                "confidence": round(conf, 4),
                "evidence": "; ".join(evidence_parts),
                "onset_time": float(onset),
            }
        return None

    def _detect_ekf_failure(self) -> dict | None:
        sv_max = _safe_get(self.features, "ekf_sv_max", 0.0)
        fault_count = _safe_get(self.features, "ekf_fault_count", 0)
        threshold = 1.0

        if sv_max > threshold or fault_count > 0:
            conf = _confidence_above(sv_max, threshold, threshold)
            if fault_count > 0:
                conf = max(conf, 0.6 + min(0.4, fault_count * 0.1))
            conf = min(1.0, conf)
            onset = _safe_get(self.features, "ekf_sv_onset_time", 0.0)
            evidence_parts = []
            if sv_max > threshold:
                evidence_parts.append(f"ekf_sv_max={sv_max:.2f}")
            if fault_count > 0:
                evidence_parts.append(f"ekf_fault_count={fault_count}")
            return {
                "failure_class": "ekf_failure",
                "confidence": round(conf, 4),
                "evidence": "; ".join(evidence_parts),
                "onset_time": float(onset),
            }
        return None

    def _detect_compass_interference(self) -> dict | None:
        mag_range = _safe_get(self.features, "mag_field_range", 0.0)
        interference = _safe_get(self.features, "mag_interference_likely", False)
        threshold = 200.0

        if mag_range > threshold or interference:
            conf = _confidence_above(mag_range, threshold, threshold)
            if interference and conf < 0.6:
                conf = 0.6
            onset = _safe_get(self.features, "mag_onset_time", 0.0)
            evidence_parts = []
            if mag_range > threshold:
                evidence_parts.append(f"mag_field_range={mag_range:.1f}")
            if interference:
                evidence_parts.append("mag_interference_likely=True")
            return {
                "failure_class": "compass_interference",
                "confidence": round(conf, 4),
                "evidence": "; ".join(evidence_parts),
                "onset_time": float(onset),
            }
        return None

    def _detect_gps_glitch(self) -> dict | None:
        hdop_max = _safe_get(self.features, "gps_hdop_max", 0.0)
        nsats_min = _safe_get(self.features, "gps_nsats_min", 99)
        err_gps = _safe_get(self.features, "err_has_gps_glitch", False)
        hdop_threshold = 3.0
        nsats_threshold = 6

        triggered = hdop_max > hdop_threshold or nsats_min < nsats_threshold or err_gps
        if not triggered:
            return None

        conf = 0.0
        evidence_parts = []
        if hdop_max > hdop_threshold:
            conf = max(conf, _confidence_above(hdop_max, hdop_threshold, hdop_threshold))
            evidence_parts.append(f"gps_hdop_max={hdop_max:.1f}")
        if nsats_min < nsats_threshold:
            conf = max(conf, _confidence_below(nsats_min, nsats_threshold, nsats_threshold))
            evidence_parts.append(f"gps_nsats_min={nsats_min}")
        if err_gps:
            conf = max(conf, 0.7)
            evidence_parts.append("err_has_gps_glitch=True")
        return {
            "failure_class": "gps_glitch",
            "confidence": round(min(1.0, conf), 4),
            "evidence": "; ".join(evidence_parts),
            "onset_time": float(_safe_get(self.features, "gps_onset_time", 0.0)),
        }

    def _detect_motor_imbalance(self) -> dict | None:
        spread = _safe_get(self.features, "motor_spread_max", 0.0)
        threshold = 300.0

        if spread > threshold:
            conf = _confidence_above(spread, threshold, threshold)
            onset = _safe_get(self.features, "motor_spread_onset_time", 0.0)
            return {
                "failure_class": "motor_imbalance",
                "confidence": round(conf, 4),
                "evidence": f"motor_spread_max={spread:.0f}",
                "onset_time": float(onset),
            }
        return None

    def _detect_motor_failure(self) -> dict | None:
        sat_high = _safe_get(self.features, "motor_saturation_high_count", 0)
        sat_low = _safe_get(self.features, "motor_saturation_low_count", 0)

        if sat_high > 0 and sat_low > 0:
            conf = min(1.0, 0.5 + 0.1 * (sat_high + sat_low))
            onset = _safe_get(self.features, "motor_failure_onset_time", 0.0)
            return {
                "failure_class": "motor_failure",
                "confidence": round(conf, 4),
                "evidence": (
                    f"motor_saturation_high_count={sat_high}; "
                    f"motor_saturation_low_count={sat_low}"
                ),
                "onset_time": float(onset),
            }
        return None

    def _detect_power_issue(self) -> dict | None:
        volt_min = _safe_get(self.features, "bat_volt_min", 99.0)
        drop_rate = _safe_get(self.features, "bat_max_drop_rate", 0.0)
        volt_threshold = 10.5
        drop_threshold = -1.0

        triggered = volt_min < volt_threshold or drop_rate < drop_threshold
        if not triggered:
            return None

        conf = 0.0
        evidence_parts = []
        if volt_min < volt_threshold:
            conf = max(conf, _confidence_below(volt_min, volt_threshold, volt_threshold))
            evidence_parts.append(f"bat_volt_min={volt_min:.2f}")
        if drop_rate < drop_threshold:
            conf = max(conf, min(1.0, abs(drop_rate - drop_threshold) / abs(drop_threshold)))
            evidence_parts.append(f"bat_max_drop_rate={drop_rate:.2f}")
        return {
            "failure_class": "power_issue",
            "confidence": round(min(1.0, conf), 4),
            "evidence": "; ".join(evidence_parts),
            "onset_time": float(_safe_get(self.features, "power_onset_time", 0.0)),
        }

    def _detect_rc_failsafe(self) -> dict | None:
        rc_fs = _safe_get(self.features, "rc_likely_failsafe", False)
        err_sub2 = _safe_get(self.features, "err_subsystem_2", False)
        err_sub5 = _safe_get(self.features, "err_subsystem_5", False)

        if rc_fs or err_sub2 or err_sub5:
            conf = 0.8
            evidence_parts = []
            if rc_fs:
                evidence_parts.append("rc_likely_failsafe=True")
                conf = 0.9
            if err_sub2:
                evidence_parts.append("err_subsystem_2=True")
            if err_sub5:
                evidence_parts.append("err_subsystem_5=True")
            return {
                "failure_class": "rc_failsafe",
                "confidence": round(min(1.0, conf), 4),
                "evidence": "; ".join(evidence_parts),
                "onset_time": float(_safe_get(self.features, "rc_failsafe_onset_time", 0.0)),
            }
        return None

    def _detect_thrust_loss(self) -> dict | None:
        sat_time = _safe_get(self.features, "motor_saturation_high_time", 0.0)
        threshold = 3.0

        if sat_time > threshold:
            conf = _confidence_above(sat_time, threshold, threshold)
            onset = _safe_get(self.features, "thrust_loss_onset_time", 0.0)
            return {
                "failure_class": "thrust_loss",
                "confidence": round(conf, 4),
                "evidence": f"motor_saturation_high_time={sat_time:.1f}s",
                "onset_time": float(onset),
            }
        return None

    def _detect_tuning_issue(self) -> dict | None:
        stock_pids = _safe_get(self.features, "stock_pids_detected", False)
        att_err = _safe_get(self.features, "attitude_error_max", 0.0)
        att_threshold = 10.0  # degrees

        if stock_pids and att_err > att_threshold:
            conf = _confidence_above(att_err, att_threshold, att_threshold)
            conf = max(conf, 0.5)
            return {
                "failure_class": "tuning_issue",
                "confidence": round(conf, 4),
                "evidence": (
                    f"stock_pids_detected=True; attitude_error_max={att_err:.1f}"
                ),
                "onset_time": float(_safe_get(self.features, "tuning_onset_time", 0.0)),
            }
        return None

    def _detect_mechanical_failure(self) -> dict | None:
        mech = _safe_get(self.features, "mechanical_failure_detected", False)
        if mech:
            conf = _safe_get(self.features, "mechanical_failure_confidence", 0.7)
            return {
                "failure_class": "mechanical_failure",
                "confidence": round(min(1.0, conf), 4),
                "evidence": "mechanical_failure_detected=True",
                "onset_time": float(
                    _safe_get(self.features, "mechanical_failure_onset_time", 0.0)
                ),
            }
        return None

    def _detect_esc_desync(self) -> dict | None:
        desync = _safe_get(self.features, "esc_desync_detected", False)
        if desync:
            conf = _safe_get(self.features, "esc_desync_confidence", 0.7)
            return {
                "failure_class": "esc_desync",
                "confidence": round(min(1.0, conf), 4),
                "evidence": "esc_desync_detected=True",
                "onset_time": float(
                    _safe_get(self.features, "esc_desync_onset_time", 0.0)
                ),
            }
        return None

    def _detect_brownout(self) -> dict | None:
        vcc_min = _safe_get(self.features, "powr_vcc_min", 5.0)
        threshold = 4.5

        if vcc_min < threshold:
            conf = _confidence_below(vcc_min, threshold, threshold)
            return {
                "failure_class": "brownout",
                "confidence": round(conf, 4),
                "evidence": f"powr_vcc_min={vcc_min:.2f}V",
                "onset_time": float(
                    _safe_get(self.features, "brownout_onset_time", 0.0)
                ),
            }
        return None

    def _detect_scheduling_overrun(self) -> dict | None:
        overrun = _safe_get(self.features, "pm_scheduling_overrun", False)
        nlon = _safe_get(self.features, "pm_nlon_total", 0)
        nlon_threshold = 10

        if overrun or nlon > nlon_threshold:
            conf = 0.6
            evidence_parts = []
            if overrun:
                evidence_parts.append("pm_scheduling_overrun=True")
                conf = 0.7
            if nlon > nlon_threshold:
                conf = max(conf, _confidence_above(nlon, nlon_threshold, nlon_threshold))
                evidence_parts.append(f"pm_nlon_total={nlon}")
            return {
                "failure_class": "scheduling_overrun",
                "confidence": round(min(1.0, conf), 4),
                "evidence": "; ".join(evidence_parts),
                "onset_time": float(
                    _safe_get(self.features, "scheduling_onset_time", 0.0)
                ),
            }
        return None

    # --- main entry point ---

    def classify_rule_based(self) -> list[dict]:
        """Run all detection rules, return sorted detections.

        Returns detections sorted by confidence (descending). If nothing
        fires, returns a single "normal" entry.
        """
        detectors = [
            self._detect_vibration_high,
            self._detect_ekf_failure,
            self._detect_compass_interference,
            self._detect_gps_glitch,
            self._detect_motor_imbalance,
            self._detect_motor_failure,
            self._detect_power_issue,
            self._detect_rc_failsafe,
            self._detect_thrust_loss,
            self._detect_tuning_issue,
            self._detect_mechanical_failure,
            self._detect_esc_desync,
            self._detect_brownout,
            self._detect_scheduling_overrun,
        ]

        detections: list[dict] = []
        for detect_fn in detectors:
            result = detect_fn()
            if result is not None:
                detections.append(result)

        if not detections:
            detections.append(
                {
                    "failure_class": "normal",
                    "confidence": 1.0,
                    "evidence": "No anomalies detected",
                    "onset_time": 0.0,
                }
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    # --- causal arbiter ---

    def causal_arbiter(self, detections: list[dict]) -> list[dict]:
        """Annotate detections with causal relationships.

        The earliest anomaly not explained by an even-earlier one gets
        marked as root cause. Later anomalies in known causal chains
        from the root cause get marked as downstream.

        Adds to each dict: is_root_cause, is_downstream, caused_by.
        """
        if not detections:
            return detections

        anomalies = [d for d in detections if d["failure_class"] != "normal"]
        normals = [d for d in detections if d["failure_class"] == "normal"]

        if not anomalies:
            for d in normals:
                d["is_root_cause"] = False
                d["is_downstream"] = False
                d["caused_by"] = None
            return detections

        # sort by onset for temporal ordering
        anomalies.sort(key=lambda d: d["onset_time"])

        # NB: build reverse lookup so we can check if an effect has a known cause
        effect_to_causes: dict[str, set[str]] = {}
        for cause, effect in CAUSAL_CHAINS:
            effect_to_causes.setdefault(effect, set()).add(cause)

        identified_causes: set[str] = set()
        root_cause_class: str | None = None

        for d in anomalies:
            d["is_root_cause"] = False
            d["is_downstream"] = False
            d["caused_by"] = None

        # first pass: find root cause (earliest unexplained anomaly)
        for d in anomalies:
            fc = d["failure_class"]
            possible_causes = effect_to_causes.get(fc, set())
            explained_by = possible_causes & identified_causes
            if explained_by:
                d["is_downstream"] = True
                # pick the earliest matching cause
                d["caused_by"] = next(
                    a["failure_class"]
                    for a in anomalies
                    if a["failure_class"] in explained_by
                )
            else:
                if root_cause_class is None:
                    d["is_root_cause"] = True
                    root_cause_class = fc
            identified_causes.add(fc)

        for d in normals:
            d["is_root_cause"] = False
            d["is_downstream"] = False
            d["caused_by"] = None

        return detections

    # --- ML stub ---

    def classify_ml(self, model_path: str | None = None) -> list[dict]:
        """Classify using a trained XGBoost model.

        Returns predictions in the same format as classify_rule_based,
        or an empty list if no model is available.
        """
        if model_path is None or not os.path.isfile(model_path):
            return []

        try:
            import xgboost as xgb  # noqa: F811
            import numpy as np

            model = xgb.XGBClassifier()
            model.load_model(model_path)

            feature_values = np.array(
                [float(v) for v in self.features.values() if isinstance(v, (int, float))]
            ).reshape(1, -1)

            pred = model.predict(feature_values)
            proba = model.predict_proba(feature_values)

            predicted_class = FAILURE_CLASSES[int(pred[0])]
            confidence = float(proba[0][int(pred[0])])

            return [
                {
                    "failure_class": predicted_class,
                    "confidence": round(confidence, 4),
                    "evidence": "ML model prediction",
                    "onset_time": 0.0,
                }
            ]
        except Exception:
            return []
