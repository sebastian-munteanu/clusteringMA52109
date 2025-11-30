###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd
import os


def export_to_csv(
    data: pd.DataFrame,
    filename: str,
    delimiter: str = ",",
    include_index: bool = False,
) -> None:
    """
    Export a DataFrame to CSV.

    Parameters
    ----------
    data : pandas.DataFrame
    filename : str
        Output filename.
    delimiter : str, default ","
    include_index : bool, default False
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    data.to_csv(filename, sep=delimiter, index=include_index)


def export_formatted(
    data: pd.DataFrame,
    file: Union[str, TextIO],
    include_index: bool = False,
) -> None:
    """
    Export a DataFrame as a formatted text table.

    Parameters
    ----------
    data : pandas.DataFrame
    file : str or file-like
        Filename or open file handle.
    include_index : bool, default False
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    table_str = data.to_string(index=include_index)

    if isinstance(file, str):
        with open(file, "w", encoding="utf-8") as f:
            f.write(table_str)
    else:
        file.write(table_str)

def export_summary_to_csv(summary_df: pd.DataFrame, output_csv_path: str) -> None:
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    summary_df.to_csv(output_csv_path, index=False)


def export_summary_to_text(summary_df: pd.DataFrame, output_text_path: str) -> None:
    output_dir = os.path.dirname(output_text_path)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    with open(output_text_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NUMERIC COLUMN SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        for _, row in summary_df.iterrows():
            f.write(f"Column: {row['column']}\n")
            f.write(f"  Mean:          {row['mean']:.6f}\n")
            f.write(f"  Std Dev:       {row['std']:.6f}\n")
            f.write(f"  Min:           {row['min']:.6f}\n")
            f.write(f"  Max:           {row['max']:.6f}\n")
            f.write(f"  Missing Count: {int(row['missing_count'])}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")