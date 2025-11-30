###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import compute_numeric_summary


class TestDataFrameBuilder(unittest.TestCase):
    def test_define_dataframe_structure_basic(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 1.0, 2.0]},
            {"name": "y", "reps": [10.0, 11.0, 12.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        self.assertEqual(seed_df.shape, (3, 2))
        self.assertListEqual(list(seed_df.columns), ["x", "y"])
        self.assertTrue(np.allclose(seed_df["x"].values, [0.0, 1.0, 2.0]))

    def test_simulate_data_shape(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 5.0]},
            {"name": "y", "reps": [2.0, 4.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        data = simulate_data(seed_df, n_points=100, random_state=1)
        self.assertEqual(data.shape[0], 100)
        self.assertIn("true_cluster", data.columns)

    def test_compute_numeric_summary(self):
        """Test compute_numeric_summary with mixed columns and missing values."""
        # Create test DataFrame: 3 numeric cols, 1 non-numeric, at least 1 missing value
        test_df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, np.nan],
            'y': [10.0, 20.0, np.nan, 40.0],
            'z': [100.0, 200.0, 300.0, 400.0],
            'label': ['A', 'B', 'C', 'D'],  # non-numeric column
        })

        summary = compute_numeric_summary(test_df)

        # Check shape and columns
        self.assertEqual(summary.shape[0], 3)  # 3 numeric columns
        self.assertListEqual(
            list(summary.columns),
            ['column', 'mean', 'std', 'min', 'max', 'missing_count']
        )

        # Check column names
        self.assertListEqual(list(summary['column']), ['x', 'y', 'z'])

        # Check 'x' column (1 missing value)
        x_row = summary[summary['column'] == 'x'].iloc[0]
        self.assertAlmostEqual(x_row['mean'], 2.0, places=5)
        self.assertAlmostEqual(x_row['min'], 1.0, places=5)
        self.assertAlmostEqual(x_row['max'], 3.0, places=5)
        self.assertEqual(x_row['missing_count'], 1)

        # Check 'y' column (1 missing value)
        y_row = summary[summary['column'] == 'y'].iloc[0]
        self.assertAlmostEqual(y_row['mean'], 23.333333, places=4)
        self.assertEqual(y_row['missing_count'], 1)

        # Check 'z' column (no missing values)
        z_row = summary[summary['column'] == 'z'].iloc[0]
        self.assertAlmostEqual(z_row['mean'], 250.0, places=5)
        self.assertEqual(z_row['missing_count'], 0)


if __name__ == "__main__":
    unittest.main()