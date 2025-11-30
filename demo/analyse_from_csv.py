###
## cluster_maker: demo for data analysis and export
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import sys
import os
import pandas as pd

from cluster_maker.data_analyser import compute_numeric_summary
from cluster_maker.data_exporter import export_summary_to_csv, export_summary_to_text

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    """
    Analyse a CSV file and export numeric summary statistics.

    Parameters
    ----------
    args : list[str]
        Command-line arguments (sys.argv).
        Expected: [script_name, input_csv_path]
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)

    print("=== cluster_maker demo: data analysis and export ===\n")

    # Validate command-line arguments
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(0)

    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(0)

    # Read CSV file
    try:
        df = pd.read_csv(input_path)
        print(f"CSV file read successfully.")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as exc:
        print(f"ERROR: Failed to read CSV file: {exc}")
        sys.exit(0)

    # Compute numeric summary
    try:
        summary = compute_numeric_summary(df)
        if summary.shape[0] == 0:
            print("WARNING: No numeric columns found in the input file.")
        else:
            print(f"Summary computed for {summary.shape[0]} numeric column(s).")
    except Exception as exc:
        print(f"ERROR: Failed to compute summary: {exc}")
        sys.exit(0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Export summary to CSV
    output_csv_path = os.path.join(output_dir, "summary.csv")
    print(f"\nExporting summary to CSV: {output_csv_path}")
    try:
        export_summary_to_csv(summary, output_csv_path)
    except Exception as exc:
        print(f"ERROR: Failed to save CSV file: {exc}")
        sys.exit(0)

    # Export summary to text file
    output_text_path = os.path.join(output_dir, "summary.txt")
    print(f"Exporting summary to text: {output_text_path}")
    try:
        export_summary_to_text(summary, output_text_path)
    except Exception as exc:
        print(f"ERROR: Failed to save text file: {exc}")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv)