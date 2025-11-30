###
## cluster_maker demo: PCA and cluster diagnostics
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.dimensionality_reduction import apply_pca, pca_explained_variance_summary
from cluster_maker.evaluation import silhouette_analysis
from cluster_maker.algorithms import kmeans


def main():
    print("=== cluster_maker demo: PCA and cluster diagnostics ===\n")

    # Generate synthetic data
    np.random.seed(42)
    n_per_cluster = 50
    
    # Create 3 clusters in 5D space
    c1 = np.random.randn(n_per_cluster, 5) + np.array([0, 0, 0, 0, 0])
    c2 = np.random.randn(n_per_cluster, 5) + np.array([5, 5, 5, 5, 5])
    c3 = np.random.randn(n_per_cluster, 5) + np.array([-5, -5, -5, -5, -5])
    X = np.vstack([c1, c2, c3])
    true_labels = np.array([0]*n_per_cluster + [1]*n_per_cluster + [2]*n_per_cluster)

    print(f"Generated synthetic data: {X.shape}")
    print(f"  3 clusters, 5 dimensions, {n_per_cluster} samples per cluster\n")

    # Apply PCA
    print("Applying PCA with variance threshold = 0.95...")
    X_pca, pca_model = apply_pca(X, variance_threshold=0.95)
    print(f"✓ Reduced to {pca_model.n_components_} components")
    
    # Variance summary
    var_summary = pca_explained_variance_summary(pca_model)
    print("\nExplained variance by component:")
    print(var_summary.to_string(index=False))

    # Cluster using K-means on original data
    print(f"\nApplying K-means clustering (k=3)...")
    labels, centroids = kmeans(X, k=3, random_state=42)
    print("✓ Clustering complete")

    # Silhouette analysis
    print(f"\nComputing silhouette scores...")
    silhouette_result = silhouette_analysis(X, labels)
    print(f"  Mean silhouette score: {silhouette_result['silhouette_mean']:.4f}")
    print(f"  Assessment: {silhouette_result['quality_assessment']}")
    print("\n  Per-cluster scores:")
    for cluster_id, score in silhouette_result['silhouette_by_cluster'].items():
        print(f"    Cluster {cluster_id}: {score:.4f}")

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Original dimensions: {X.shape[1]}")
    print(f"  Reduced dimensions: {X_pca.shape[1]}")
    print(f"  Variance retained: {var_summary['cumulative_variance'].iloc[-1]:.2%}")
    print(f"  Cluster quality (silhouette): {silhouette_result['silhouette_mean']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()