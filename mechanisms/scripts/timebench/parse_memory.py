#!/usr/bin/env python3
"""
Parse memory logs from FHAIM L1 and L2 runs and display a summary table.
"""

import re
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent.parent.parent / "data" / "logs" / "run_42"

DATASETS = ["breast_train", "compas_train", "diabetes_train"]
METHODS = ["fhaim_l1", "fhaim_l2"]

# Display names for better table formatting
DATASET_NAMES = {
    "breast_train": "Breast Cancer",
    "compas_train": "COMPAS",
    "diabetes_train": "Diabetes",
}

METHOD_NAMES = {
    "fhaim_l1": "FHAIM-L1",
    "fhaim_l2": "FHAIM-L2",
}


def parse_log_file(log_path):
    """Parse a single log file and extract memory information."""
    if not log_path.exists():
        return None

    with open(log_path, "r") as f:
        content = f.read()

    results = {
        "current_mb": None,
        "peak_mb": None,
    }

    # Parse memory values
    match = re.search(r"\[MEMORY\] Current: ([\d.]+) MB", content)
    if match:
        results["current_mb"] = float(match.group(1))

    match = re.search(r"\[MEMORY\] Peak: ([\d.]+) MB", content)
    if match:
        results["peak_mb"] = float(match.group(1))

    return results


def print_table():
    """Print the memory summary table."""
    print("=" * 70)
    print("FHAIM MEMORY BENCHMARK (run_42)")
    print("=" * 70)
    print()

    # Collect all data first
    data = {}
    for dataset in DATASETS:
        data[dataset] = {}
        for method in METHODS:
            log_file = LOG_DIR / dataset / f"{method}_eps_1.0.log"
            data[dataset][method] = parse_log_file(log_file)

    # Print table header
    print(f"{'Dataset':<15} | {'FHAIM-L1':>12} | {'FHAIM-L2':>12} |")
    print("-" * 70)

    # Print peak memory table
    print("\nPeak Memory (MB):")
    print("-" * 45)
    print(f"{'Dataset':<15} | {'FHAIM-L1':>12} | {'FHAIM-L2':>12} |")
    print("-" * 45)

    for dataset in DATASETS:
        row = f"{DATASET_NAMES[dataset]:<15} |"
        for method in METHODS:
            result = data[dataset][method]
            if result and result["peak_mb"] is not None:
                row += f" {result['peak_mb']:>10.2f} MB |"
            else:
                row += f" {'N/A':>12} |"
        print(row)

    print("-" * 45)

    # Print current memory table
    print("\nCurrent Memory at End (MB):")
    print("-" * 45)
    print(f"{'Dataset':<15} | {'FHAIM-L1':>12} | {'FHAIM-L2':>12} |")
    print("-" * 45)

    for dataset in DATASETS:
        row = f"{DATASET_NAMES[dataset]:<15} |"
        for method in METHODS:
            result = data[dataset][method]
            if result and result["current_mb"] is not None:
                row += f" {result['current_mb']:>10.2f} MB |"
            else:
                row += f" {'N/A':>12} |"
        print(row)

    print("-" * 45)
    print()

    # Print LaTeX table
    print_latex_table(data)


def print_latex_table(data):
    """Print LaTeX formatted table."""
    print("\n" + "=" * 70)
    print("LaTeX Table:")
    print("=" * 70)
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Peak Memory Usage (MB)}")
    print(r"\label{tab:memory}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Dataset & FHAIM-L1 & FHAIM-L2 \\")
    print(r"\midrule")

    for dataset in DATASETS:
        row = f"{DATASET_NAMES[dataset]}"
        for method in METHODS:
            result = data[dataset][method]
            if result and result["peak_mb"] is not None:
                row += f" & {result['peak_mb']:.2f}"
            else:
                row += " & N/A"
        row += r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    print_table()
