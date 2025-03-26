"""Automated dataset generation by running SITL scenarios."""

import argparse
import csv
import logging
import os
import shutil
import time
from typing import Sequence

from flight_analyst.sitl.inject import SITLInjector
from flight_analyst.sitl.scenarios import SCENARIOS, get_scenario, list_scenarios  # noqa: F401

logger = logging.getLogger(__name__)

# fault type -> injector method name
_FAULT_METHOD_MAP = {
    "motor_failure": "inject_motor_failure",
    "gps_glitch": "inject_gps_glitch",
    "gps_loss": "inject_gps_loss",
    "compass_interference": "inject_compass_interference",
    "compass_failure": "inject_compass_failure",
    "battery_drop": "inject_battery_drop",
    "vibration": "inject_vibration",
    "rc_failsafe": "inject_rc_failsafe",
    "imu_bias": "inject_imu_bias",
    "wind": "inject_wind",
}


def _apply_fault(injector: SITLInjector, fault: dict) -> None:
    fault_type = fault["type"]
    method_name = _FAULT_METHOD_MAP.get(fault_type)
    if method_name is None:
        raise ValueError(f"Unknown fault type: {fault_type}")

    method = getattr(injector, method_name)
    kwargs = {k: v for k, v in fault.items() if k != "type"}
    method(**kwargs)


def run_scenario(
    injector: SITLInjector,
    scenario: dict,
    log_dir: str = "dataset",
) -> dict | None:
    """Run a single scenario: arm, takeoff, inject fault, observe, land.

    Returns a metadata dict or None if log collection failed.
    """
    setup = scenario["setup"]
    target_alt = setup["target_alt"]
    hover_time = setup["hover_time"]
    mode = setup["mode"]

    extra_params = setup.get("params", {})
    for pname, pval in extra_params.items():
        injector.set_param(pname, pval)

    injector.change_mode("GUIDED")
    injector.arm()
    injector.takeoff(target_alt=target_alt)
    injector.change_mode(mode)

    logger.info("Hovering for %d s before fault injection ...", hover_time)
    time.sleep(hover_time)

    flight_start = time.monotonic()

    fault = scenario.get("fault")
    if fault is not None:
        logger.info("Injecting fault: %s", fault.get("type"))
        _apply_fault(injector, fault)

    observe_time = scenario.get("observe_time", 0)
    if observe_time > 0:
        logger.info("Observing for %d s ...", observe_time)
        time.sleep(observe_time)

    duration = time.monotonic() - flight_start

    injector.clear_all_faults()
    try:
        injector.change_mode("LAND")
        injector.wait_disarmed(timeout=120)
    except TimeoutError:
        logger.warning("Disarm timeout; forcing disarm")
        injector.disarm(force=True)

    log_path = injector.get_latest_log(log_dir=log_dir)
    if log_path is None:
        logger.warning("No log file found in %s", log_dir)
        return None

    return {
        "log_path": log_path,
        "label": scenario["label"],
        "scenario": scenario.get("description", ""),
        "description": scenario.get("description", ""),
        "duration": round(duration, 1),
    }


def generate_dataset(
    scenarios: Sequence[str] | None = None,
    repeats: int = 3,
    output_dir: str = "dataset",
    connection_string: str = "tcp:127.0.0.1:5760",
) -> str:
    """Run selected scenarios, collect logs, write manifest.csv.

    Returns path to the generated manifest.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    scenario_names = scenarios if scenarios is not None else list_scenarios()
    logger.info(
        "Generating dataset: %d scenarios x %d repeats -> %s",
        len(scenario_names),
        repeats,
        output_dir,
    )

    injector = SITLInjector(connection_string=connection_string)
    injector.connect()

    rows: list[dict] = []

    for name in scenario_names:
        scenario = get_scenario(name)
        for run_idx in range(repeats):
            tag = f"{name}_run{run_idx}"
            logger.info("--- %s (%d/%d) ---", name, run_idx + 1, repeats)

            try:
                meta = run_scenario(injector, scenario, log_dir=output_dir)
            except Exception:
                logger.exception("Scenario %s run %d failed", name, run_idx)
                meta = None

            if meta is not None:
                dest = os.path.join(output_dir, f"{tag}.BIN")
                if meta["log_path"] != dest:
                    shutil.copy2(meta["log_path"], dest)
                meta["log_path"] = dest
                rows.append(meta)

    fieldnames = ["log_path", "label", "scenario", "description", "duration"]
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Manifest written to %s (%d rows)", manifest_path, len(rows))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labelled SITL flight-log dataset",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names to run (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repetitions per scenario (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="dataset",
        help="Output directory (default: dataset)",
    )
    parser.add_argument(
        "--connection",
        default="tcp:127.0.0.1:5760",
        help="MAVLink connection string (default: tcp:127.0.0.1:5760)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    generate_dataset(
        scenarios=args.scenarios,
        repeats=args.repeats,
        output_dir=args.output,
        connection_string=args.connection,
    )


if __name__ == "__main__":
    main()
