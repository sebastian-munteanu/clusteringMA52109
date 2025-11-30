###
## cluster_maker: PCA-based dimensionality reduction
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(
    X: np.ndarray | pd.DataFrame,
    n_components: int | None = None,
    variance_threshold: float | None = None,
) -> tuple[np.ndarray, PCA]:
    """
    Apply Principal Component Analysis (PCA) to reduce dimensionality.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input data (n_samples, n_features).
    n_components : int, optional
        Number of components to retain. If None, use variance_threshold.
    variance_threshold : float, optional
        Cumulative variance threshold (0.0–1.0). Determines n_components
        automatically if n_components is None. Default: 0.95 (95% variance).

    Returns
    -------
    X_reduced : np.ndarray
        Transformed data in PCA space (n_samples, n_components_used).
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model for inverse transform or inspection.

    Raises
    ------
    ValueError
        If n_components and variance_threshold are both None, or if
        variance_threshold is not in (0, 1].

    Notes
    -----
    - If both n_components and variance_threshold are provided, n_components
      takes precedence.
    - Automatically centers data (zero mean, unit variance).
    """
    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError("Input X must be 2-dimensional.")
    if X_array.shape[0] < 2:
        raise ValueError("Input X must have at least 2 samples.")

    if n_components is None and variance_threshold is None:
        variance_threshold = 0.95

    if variance_threshold is not None:
        if not (0.0 < variance_threshold <= 1.0):
            raise ValueError("variance_threshold must be in (0.0, 1.0].")
        # Fit with all components first to determine cutoff
        pca_full = PCA()
        pca_full.fit(X_array)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = np.argmax(cumsum >= variance_threshold) + 1
        n_components = min(n_comp, X_array.shape[1])

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_array)

    return X_reduced, pca


def pca_explained_variance_summary(pca_model: PCA) -> pd.DataFrame:
    """
    Summarize explained variance for each principal component.

    Parameters
    ----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame with columns:
        - 'component': PC index (1, 2, ...)
        - 'explained_variance': variance explained by this PC
        - 'cumulative_variance': cumulative variance up to this PC
    """
    n_comp = pca_model.n_components_
    explained_var = pca_model.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)

    return pd.DataFrame({
        'component': np.arange(1, n_comp + 1),
        'explained_variance': explained_var,
        'cumulative_variance': cumsum_var,
    })