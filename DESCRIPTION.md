# cluster_maker

## Overview
**cluster_maker** is a Python package for clustering analysis that provides a complete workflow for preparing data, applying clustering algorithms, evaluating results, and visualizing clusters. It abstracts away common preprocessing, standardization, and evaluation steps, allowing users to focus on analyzing clustered data.

## Main Components

### 1. **dataframe_builder.py**
Defines the core data structures for clustering:
- `define_dataframe_structure()`: Creates a seed DataFrame describing cluster centres (rows = clusters, columns = features).
- `simulate_data()`: Generates synthetic clustered data around specified centres with configurable noise (Gaussian perturbation).

### 2. **preprocessing.py**
Handles data preparation and transformation:
- Feature selection and validation
- Data standardization (e.g., z-score normalization)
- Handling missing values
- Ensures data is ready for clustering algorithms

### 3. **algorithms.py**
Implements clustering methods:
- K-means clustering (primary method)
- Extensible interface for additional algorithms
- Configurable parameters (e.g., number of clusters, random state)

### 4. **evaluation.py**
Computes clustering quality metrics:
- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index
- Inertia and elbow point detection
- Helps assess cluster validity and optimal cluster count

### 5. **plotting_clustered.py**
Visualizes clustering results:
- 2D scatter plots of clustered data with colour-coded labels
- Elbow plots for determining optimal cluster count
- Matplotlib-based rendering

### 6. **data_analyser.py**
(Task 3a) Provides statistical summary functions:
- Computes descriptive statistics (mean, std, min, max, missing count) for numeric columns
- Generates a summary DataFrame

### 7. **data_exporter.py**
(Task 3b) Exports analysis results:
- Saves summary DataFrames to CSV
- Exports human-readable text reports

### 8. **interface.py**
High-level orchestration function:
- `run_clustering()`: Single entry point combining preprocessing, clustering, evaluation, and visualization
- Handles input validation and error reporting

## Workflow

Typical usage follows this pipeline:

```
Input CSV → Validate features → Standardize data → Apply clustering algorithm
    → Compute metrics → Generate visualizations → Export results
```

## Key Features

- **Modular design**: Each step (preprocessing, clustering, evaluation, visualization) is independent
- **Flexible input**: Accepts CSV files with user-specified feature columns
- **Robust error handling**: Clear, controlled exceptions for invalid inputs
- **Metrics and diagnostics**: Multiple clustering quality measures
- **Visualization**: Automatic 2D plots and elbow curves

## Allowed Libraries

- numpy
- pandas
- scikit-learn (sklearn)
- matplotlib

## Example Usage

```python
from cluster_maker import run_clustering

result = run_clustering(
    input_path="data/my_data.csv",
    feature_cols=["x", "y"],
    algorithm="kmeans",
    k=3,
    standardise=True,
    output_path="output/clustered_data.csv",
    random_state=42,
    compute_elbow=True,
)

print(result["metrics"])
result["fig_cluster"].show()
```