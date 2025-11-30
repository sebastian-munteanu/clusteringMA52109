###
## cluster_maker - test file for dimensionality reduction
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from cluster_maker.dimensionality_reduction import apply_pca, pca_explained_variance_summary
from cluster_maker.evaluation import silhouette_analysis


class TestPCABasic(unittest.TestCase):
    """Test basic PCA functionality."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        # Generate 100 samples in 5D space, but with strong correlation
        # (intrinsically lower-dimensional)
        n_samples = 100
        self.X = np.random.randn(n_samples, 5)
        # Add correlation: dim 2 ≈ dim 1, dim 4 ≈ dim 3
        self.X[:, 1] = self.X[:, 0] + 0.1 * np.random.randn(n_samples)
        self.X[:, 4] = self.X[:, 3] + 0.1 * np.random.randn(n_samples)

    def test_pca_n_components_fixed(self):
        """Test PCA with fixed n_components."""
        X_reduced, pca = apply_pca(self.X, n_components=2)

        # Check shape
        self.assertEqual(X_reduced.shape, (100, 2))

        # Check PCA model
        self.assertEqual(pca.n_components_, 2)
        self.assertEqual(len(pca.explained_variance_ratio_), 2)

        # Explained variance should be positive and sum to ≤ 1.0
        self.assertTrue(np.all(pca.explained_variance_ratio_ > 0))
        self.assertLessEqual(np.sum(pca.explained_variance_ratio_), 1.0)

    def test_pca_variance_threshold(self):
        """Test PCA with variance threshold."""
        X_reduced, pca = apply_pca(self.X, variance_threshold=0.90)

        # With correlated data and threshold 0.90, should need ≤ 3 components
        self.assertLessEqual(pca.n_components_, 3)

        # Cumulative variance should exceed threshold
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        self.assertGreaterEqual(cumsum[-1], 0.90)

    def test_pca_invalid_variance_threshold(self):
        """Test that invalid variance_threshold raises ValueError."""
        with self.assertRaises(ValueError):
            apply_pca(self.X, variance_threshold=1.5)  # > 1.0

        with self.assertRaises(ValueError):
            apply_pca(self.X, variance_threshold=0.0)  # ≤ 0.0

    def test_pca_no_parameters_defaults(self):
        """Test that default variance_threshold is applied."""
        X_reduced, pca = apply_pca(self.X)

        # With default 0.95, should need some components
        self.assertGreater(pca.n_components_, 0)
        self.assertLessEqual(pca.n_components_, self.X.shape[1])

    def test_pca_dataframe_input(self):
        """Test PCA with pandas DataFrame input."""
        df = pd.DataFrame(self.X, columns=['a', 'b', 'c', 'd', 'e'])
        X_reduced, pca = apply_pca(df, n_components=3)

        self.assertEqual(X_reduced.shape, (100, 3))
        self.assertEqual(pca.n_components_, 3)


class TestPCAExplainedVariance(unittest.TestCase):
    """Test explained variance summary."""

    def setUp(self):
        """Create and fit a PCA model."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        self.X_reduced, self.pca = apply_pca(X, n_components=3)

    def test_variance_summary_shape(self):
        """Test that variance summary has correct shape."""
        summary = pca_explained_variance_summary(self.pca)

        self.assertEqual(summary.shape[0], 3)  # 3 components
        self.assertListEqual(
            list(summary.columns),
            ['component', 'explained_variance', 'cumulative_variance']
        )

    def test_variance_summary_monotonic(self):
        """Test that cumulative variance is monotonically increasing."""
        summary = pca_explained_variance_summary(self.pca)
        cumsum = summary['cumulative_variance'].values

        # Check monotonic increase
        for i in range(1, len(cumsum)):
            self.assertGreaterEqual(cumsum[i], cumsum[i-1])

    def test_variance_summary_bounds(self):
        """Test that variance values are in valid range."""
        summary = pca_explained_variance_summary(self.pca)

        # Individual variances in [0, 1]
        self.assertTrue((summary['explained_variance'] >= 0).all())
        self.assertTrue((summary['explained_variance'] <= 1).all())

        # Cumulative variances in [0, 1]
        self.assertTrue((summary['cumulative_variance'] >= 0).all())
        self.assertTrue((summary['cumulative_variance'] <= 1).all())


class TestSilhouetteAnalysis(unittest.TestCase):
    """Test silhouette-based cluster diagnostics."""

    def setUp(self):
        """Create well-separated clusters."""
        np.random.seed(42)
        # Cluster 1: centered at (0, 0)
        c1 = np.random.randn(30, 2) + np.array([0, 0])
        # Cluster 2: centered at (5, 5)
        c2 = np.random.randn(30, 2) + np.array([5, 5])
        self.X_well_separated = np.vstack([c1, c2])
        self.labels_well = np.array([0]*30 + [1]*30)

        # Poorly separated clusters (overlapping)
        c1_poor = np.random.randn(30, 2) + np.array([0, 0])
        c2_poor = np.random.randn(30, 2) + np.array([1, 1])
        self.X_poorly_separated = np.vstack([c1_poor, c2_poor])
        self.labels_poor = np.array([0]*30 + [1]*30)

    def test_silhouette_result_structure(self):
        """Test that silhouette analysis returns expected keys."""
        result = silhouette_analysis(self.X_well_separated, self.labels_well)

        expected_keys = {
            'silhouette_scores',
            'silhouette_mean',
            'silhouette_by_cluster',
            'quality_assessment',
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_silhouette_well_separated(self):
        """Test that well-separated clusters have high silhouette scores."""
        result = silhouette_analysis(self.X_well_separated, self.labels_well)

        # Mean silhouette should be high (> 0.5)
        self.assertGreater(result['silhouette_mean'], 0.5)
        self.assertIn("GOOD", result['quality_assessment'])

    def test_silhouette_poorly_separated(self):
        """Test that overlapping clusters have lower silhouette scores."""
        result = silhouette_analysis(self.X_poorly_separated, self.labels_poor)

        # Mean silhouette should be lower
        self.assertLess(result['silhouette_mean'], 0.5)

    def test_silhouette_per_cluster(self):
        """Test per-cluster silhouette scores."""
        result = silhouette_analysis(self.X_well_separated, self.labels_well)

        by_cluster = result['silhouette_by_cluster']
        self.assertEqual(len(by_cluster), 2)  # 2 clusters
        self.assertIn(0, by_cluster)
        self.assertIn(1, by_cluster)

        # Scores should be in [-1, 1]
        for score in by_cluster.values():
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_silhouette_sample_scores_shape(self):
        """Test that per-sample silhouette scores have correct shape."""
        result = silhouette_analysis(self.X_well_separated, self.labels_well)

        scores = result['silhouette_scores']
        self.assertEqual(len(scores), 60)  # 60 samples total


if __name__ == "__main__":
    unittest.main()