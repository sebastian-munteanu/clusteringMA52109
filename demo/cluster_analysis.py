###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker import run_clustering, select_features

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("=== cluster_maker demo: clustering analysis ===\n")

    # Require exactly one argument: the CSV file path
    if len(args) != 1:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/demo_cluster_analysis.py [input_csv_file]")
        sys.exit(1)

    # Input CSV file
    input_path = "../data/demo_data.csv"
    print(f"Input CSV file: {input_path}")

    # Check file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Load data so we can inspect numeric columns
    print("\nLoading data to inspect available features...")
    df = pd.read_csv(input_path)
    print("Data loaded successfully.")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Identify numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(numeric_cols) < 2:
        print("\nERROR: Not enough numeric columns for 2D clustering.")
        print(f"Numeric columns found: {numeric_cols}")
        sys.exit(1)

    # Take the first two numeric columns
    feature_cols = numeric_cols[:2]
    print(f"Chosen numeric feature columns for clustering: {feature_cols}")
    print("-" * 60)

    # Validate feature columns using the package function
    try:
        _ = select_features(df, feature_cols)
    except Exception as exc:
        print(f"\nERROR validating features with select_features:\n{exc}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run the orchestrator
    print("Running clustering with run_clustering(...)")
    result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=3,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, "clustered_data.csv"),
        random_state=42,
        compute_elbow=True,
    )

    print("\nClustering completed.")
    print("Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    print("-" * 60)

    # Save plots
    cluster_plot_path = os.path.join(OUTPUT_DIR, "cluster_plot.png")
    elbow_plot_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")

    print(f"Saving 2D cluster plot to:\n  {cluster_plot_path}")
    result["fig_cluster"].savefig(cluster_plot_path, dpi=150)

    if result["fig_elbow"] is not None:
        print(f"Saving elbow plot to:\n  {elbow_plot_path}")
        result["fig_elbow"].savefig(elbow_plot_path, dpi=150)
    else:
        print("No elbow plot was generated (fig_elbow is None).")

    print("\nClustered data saved to:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'clustered_data.csv')}")
    print("Plots saved to:")
    print(f"  - {cluster_plot_path}")
    if result["fig_elbow"] is not None:
        print(f"  - {elbow_plot_path}")

    print("\n=== End of demo ===")


if __name__ == "__main__":
    main(sys.argv)