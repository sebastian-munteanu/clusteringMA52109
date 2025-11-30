###
## cluster_maker - test file for interface and export functions
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import unittest
import tempfile
import os
import pandas as pd
import numpy as np

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_summary_to_csv, export_summary_to_text
from cluster_maker.data_analyser import compute_numeric_summary


class TestInterfaceErrorHandling(unittest.TestCase):
    """Test error handling in the high-level interface function."""

    def test_run_clustering_missing_input_file(self):
        """Test that run_clustering raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="/nonexistent/path/to/file.csv",
                feature_cols=["x", "y"],
                algorithm="kmeans",
                k=3,
            )

    def test_run_clustering_missing_feature_columns(self):
        """Test that run_clustering raises ValueError when feature columns are missing."""
        # Create a temporary CSV with columns 'a' and 'b' (not 'x' and 'y')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("a,b\n")
            tmp.write("1.0,2.0\n")
            tmp.write("3.0,4.0\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as context:
                run_clustering(
                    input_path=tmp_path,
                    feature_cols=["x", "y"],  # These columns don't exist
                    algorithm="kmeans",
                    k=3,
                )
            # Verify error message is clear
            self.assertIn("feature", str(context.exception).lower())
        finally:
            os.unlink(tmp_path)

    def test_run_clustering_insufficient_rows(self):
        """Test that run_clustering handles DataFrames with too few rows gracefully."""
        # Create a temporary CSV with only 1 row (fewer than k clusters)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("x,y\n")
            tmp.write("1.0,2.0\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises((ValueError, RuntimeError)):
                run_clustering(
                    input_path=tmp_path,
                    feature_cols=["x", "y"],
                    algorithm="kmeans",
                    k=5,  # More clusters than rows
                )
        finally:
            os.unlink(tmp_path)


class TestExportFunctionsFileCreation(unittest.TestCase):
    """Test that export functions create files correctly with valid inputs."""

    def setUp(self):
        """Create a temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()
        # Create test summary DataFrame
        self.summary_df = pd.DataFrame({
            'column': ['x', 'y', 'z'],
            'mean': [1.5, 10.5, 100.5],
            'std': [0.5, 1.5, 10.5],
            'min': [1.0, 10.0, 100.0],
            'max': [2.0, 11.0, 101.0],
            'missing_count': [0, 1, 0],
        })

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_summary_to_csv_creates_file(self):
        """Test that export_summary_to_csv creates a valid CSV file."""
        output_path = os.path.join(self.temp_dir, "summary.csv")
        export_summary_to_csv(self.summary_df, output_path)

        # Check file exists
        self.assertTrue(os.path.exists(output_path))

        # Check file can be read back
        df_read = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(df_read, self.summary_df)

    def test_export_summary_to_text_creates_file(self):
        """Test that export_summary_to_text creates a valid text file."""
        output_path = os.path.join(self.temp_dir, "summary.txt")
        export_summary_to_text(self.summary_df, output_path)

        # Check file exists
        self.assertTrue(os.path.exists(output_path))

        # Check file is not empty and contains expected text
        with open(output_path, 'r') as f:
            content = f.read()
        self.assertIn("NUMERIC COLUMN SUMMARY", content)
        self.assertIn("x", content)
        self.assertIn("y", content)
        self.assertIn("z", content)

    def test_export_summary_to_csv_invalid_directory(self):
        """Test that export_summary_to_csv raises error for non-existent directory."""
        invalid_path = "/nonexistent/directory/summary.csv"
        with self.assertRaises(ValueError) as context:
            export_summary_to_csv(self.summary_df, invalid_path)
        self.assertIn("directory", str(context.exception).lower())

    def test_export_summary_to_text_invalid_directory(self):
        """Test that export_summary_to_text raises error for non-existent directory."""
        invalid_path = "/nonexistent/directory/summary.txt"
        with self.assertRaises(ValueError) as context:
            export_summary_to_text(self.summary_df, invalid_path)
        self.assertIn("directory", str(context.exception).lower())


class TestExportWithDataAnalyser(unittest.TestCase):
    """Integration tests combining data analysis and export."""

    def setUp(self):
        """Create a temporary directory and test DataFrame."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, np.nan],
            'y': [10.0, 20.0, np.nan, 40.0],
            'z': [100.0, 200.0, 300.0, 400.0],
            'label': ['A', 'B', 'C', 'D'],
        })

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_workflow_compute_and_export(self):
        """Test complete workflow: compute summary and export to both formats."""
        # Compute summary
        summary = compute_numeric_summary(self.test_df)
        self.assertEqual(summary.shape[0], 3)  # 3 numeric columns

        # Export to CSV
        csv_path = os.path.join(self.temp_dir, "summary.csv")
        export_summary_to_csv(summary, csv_path)
        self.assertTrue(os.path.exists(csv_path))

        # Export to text
        txt_path = os.path.join(self.temp_dir, "summary.txt")
        export_summary_to_text(summary, txt_path)
        self.assertTrue(os.path.exists(txt_path))

        # Verify CSV can be read back
        df_read = pd.read_csv(csv_path)
        self.assertEqual(df_read.shape, summary.shape)

        # Verify text file contains statistics
        with open(txt_path, 'r') as f:
            content = f.read()
        self.assertIn("Missing Count:", content)


if __name__ == "__main__":
    unittest.main()