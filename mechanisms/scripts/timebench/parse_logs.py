#!/usr/bin/env python3
"""
Parse timing logs from FHAIM L1 and L2 runs and display a summary table.
"""

import os
import re
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent.parent.parent / "data" / "logs" / "run_42"

DATASETS = ["compas_train", "breast_train", "diabetes_train"]
METHODS = ["fhaim_l1", "fhaim_l2"]


def parse_log_file(log_path):
    """Parse a single log file and extract timing information."""
    if not log_path.exists():
        return None

    with open(log_path, "r") as f:
        content = f.read()

    results = {
        "compute_total": 0,
        "compute_1way": 0,
        "compute_2way": 0,
        "select_total": [],
        "select_gumbel": [],
        "measure_gaussian": [],
        "measure_oneway": [],
        "measure_step": [],
    }

    # Parse compute times
    match = re.search(r"\[TIMING\] Compute 1-way marginals: ([\d.]+)s", content)
    if match:
        results["compute_1way"] = float(match.group(1))

    match = re.search(r"\[TIMING\] Compute 2-way marginals: ([\d.]+)s", content)
    if match:
        results["compute_2way"] = float(match.group(1))

    match = re.search(r"\[TIMING\] Compute total: ([\d.]+)s", content)
    if match:
        results["compute_total"] = float(match.group(1))

    # Parse select step times (multiple occurrences)
    for match in re.finditer(r"\[TIMING\] Select step total: ([\d.]+)s", content):
        results["select_total"].append(float(match.group(1)))

    for match in re.finditer(r"\[TIMING\] Select step Gumbel noise: ([\d.]+)s", content):
        results["select_gumbel"].append(float(match.group(1)))

    # Parse measure times
    for match in re.finditer(r"\[TIMING\] Measure step \(Gaussian noise\): ([\d.]+)s", content):
        results["measure_gaussian"].append(float(match.group(1)))

    for match in re.finditer(r"\[TIMING\] Measure step: ([\d.]+)s", content):
        results["measure_step"].append(float(match.group(1)))

    match = re.search(r"\[TIMING\] One-way marginals measure total: ([\d.]+)s", content)
    if match:
        results["measure_oneway"] = float(match.group(1))

    return results


def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.2f}m"


def avg(lst):
    """Calculate average of a list."""
    return sum(lst) / len(lst) if lst else 0


def print_table():
    """Print the timing summary table."""

    # Header
    print("=" * 120)
    print("FHAIM TIMING BENCHMARK (run_42)")
    print("=" * 120)

    for method in METHODS:
        print(f"\n{'='*120}")
        print(f"  {method.upper()}")
        print("=" * 120)

        # Table header
        print(f"{'Dataset':<15} | {'Compute Total':>13} | {'1-way':>10} | {'2-way':>10} | "
              f"{'Select Avg':>11} | {'Gumbel Avg':>11} | {'Measure Avg':>11} | {'1-way Meas':>11}")
        print("-" * 120)

        for dataset in DATASETS:
            log_file = LOG_DIR / dataset / f"{method}_eps_1.0.log"
            results = parse_log_file(log_file)

            if results is None:
                print(f"{dataset:<15} | {'N/A':>13} | {'N/A':>10} | {'N/A':>10} | "
                      f"{'N/A':>11} | {'N/A':>11} | {'N/A':>11} | {'N/A':>11}")
                continue

            compute_total = format_time(results["compute_total"])
            compute_1way = format_time(results["compute_1way"])
            compute_2way = format_time(results["compute_2way"])

            select_avg = format_time(avg(results["select_total"]))
            gumbel_avg = format_time(avg(results["select_gumbel"]))
            measure_avg = format_time(avg(results["measure_gaussian"]))
            measure_oneway = format_time(results["measure_oneway"]) if isinstance(results["measure_oneway"], float) else "N/A"

            print(f"{dataset:<15} | {compute_total:>13} | {compute_1way:>10} | {compute_2way:>10} | "
                  f"{select_avg:>11} | {gumbel_avg:>11} | {measure_avg:>11} | {measure_oneway:>11}")

        print("-" * 120)

        # Print detailed stats
        print(f"\nDetailed breakdown for {method.upper()}:")
        print("-" * 80)

        for dataset in DATASETS:
            log_file = LOG_DIR / dataset / f"{method}_eps_1.0.log"
            results = parse_log_file(log_file)

            if results is None:
                continue

            print(f"\n  {dataset}:")
            print(f"    1. Compute:")
            print(f"       Total:  {format_time(results['compute_total']):>10}")
            print(f"       1-way:  {format_time(results['compute_1way']):>10}")
            print(f"       2-way:  {format_time(results['compute_2way']):>10}")

            print(f"    2. Select (avg of {len(results['select_total'])} iterations):")
            print(f"       Total:  {format_time(avg(results['select_total'])):>10}")
            print(f"       Gumbel: {format_time(avg(results['select_gumbel'])):>10}")
            print(f"       Gauss:  {format_time(avg(results['measure_gaussian'])):>10}")

            print(f"    3. Measure:")
            if results["measure_oneway"]:
                print(f"       1-way total: {format_time(results['measure_oneway']):>10}")
            if results["measure_step"]:
                print(f"       Per-step avg ({len(results['measure_step'])} steps): {format_time(avg(results['measure_step'])):>10}")


if __name__ == "__main__":
    print_table()
