###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Dict, Any, Sequence

import numpy as np
import pandas as pd


def define_dataframe_structure(column_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Define a seed DataFrame describing cluster centres.

    Parameters
    ----------
    column_specs : list of dict
        Each dict must contain:
        - 'name': str    – the column name
        - 'reps': list   – list of centre values, one per cluster

        Example:
        [
            {"name": "x", "reps": [0.0, 5.0, -5.0]},
            {"name": "y", "reps": [0.0, 5.0, -5.0]},
        ]

    Returns
    -------
    seed_df : pandas.DataFrame
        DataFrame with one row per cluster and one column per feature.
    """
    if not column_specs:
        raise ValueError("column_specs must be a non-empty list of dictionaries.")

    # Check consistency of 'reps' lengths
    reps_lengths = [len(spec.get("reps", [])) for spec in column_specs]
    if len(set(reps_lengths)) != 1:
        raise ValueError("All 'reps' lists must have the same length (number of clusters).")

    n_clusters = reps_lengths[0]
    data = {}
    for spec in column_specs:
        name = spec.get("name")
        reps = spec.get("reps")
        if name is None or reps is None:
            raise ValueError("Each column_specs entry must have 'name' and 'reps' keys.")
        if not isinstance(reps, Sequence):
            raise TypeError("'reps' must be a sequence of values.")
        if len(reps) != n_clusters:
            raise ValueError("All 'reps' lists must have the same length.")
        data[name] = list(reps)

    seed_df = pd.DataFrame.from_dict(data, orient="columns")
    seed_df.index.name = "cluster_id"
    return seed_df


def simulate_data(
    seed_df: pd.DataFrame,
    n_points: int = 100,
    cluster_std: int = 1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Simulate clustered data around the given cluster centres.

    Parameters
    ----------
    seed_df : pandas.DataFrame
        Rows represent cluster centres, columns represent features.
    n_points : int, default 100
        Total number of data points to simulate.
    cluster_std : float, default 1.0
        Standard deviation of Gaussian noise added around centres.
    random_state : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    data : pandas.DataFrame
        Simulated data with all original feature columns plus a 'true_cluster'
        column indicating the generating cluster.
    """
    if n_points <= 0:
        raise ValueError("n_points must be a positive integer.")
    if cluster_std <= 0:
        raise ValueError("cluster_std must be positive.")

    rng = np.random.RandomState(random_state)
    centres = seed_df.to_numpy(dtype=float)
    n_clusters, n_features = centres.shape

    # Distribute points as evenly as possible across clusters.
    base = n_points // n_clusters
    remainder = n_points % n_clusters
    counts = np.full(n_clusters, base, dtype=int)
    counts[:remainder] += 1

    records = []
    for cluster_id, (centre, count) in enumerate(zip(centres, counts)):
        noise = rng.normal(loc=0.0, scale=cluster_std, size=(count, n_features))
        points = centre + noise
        for point in points:
            record = {col: val for col, val in zip(seed_df.columns, point)}
            record["true_cluster"] = cluster_id
            records.append(record)

    data = pd.DataFrame.from_records(records)
    return data