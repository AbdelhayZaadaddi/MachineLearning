# Formula 1 Machine Learning Project

A collection of Jupyter notebooks applying classical machine learning techniques (clustering and classification) to the historical Formula 1 dataset. The project explores the data, then uses it both to discover natural groupings of race results (unsupervised) and to predict podium finishes (supervised).

---

## 1. Environment setup

### Requirements
- Python 3.9+
- pip
- Jupyter (Notebook or Lab)

### Create a virtual environment
```bash
cd /path/to/prj
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Install dependencies
```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

Or in one line:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### Launch Jupyter
```bash
jupyter notebook        # or: jupyter lab
```
Then open any `.ipynb` file from the browser tab that opens.

---

## 2. Dataset

All raw CSVs live in the `f1/` folder (Formula 1 World Championship dataset, 1950–2024).

| File | Description |
|------|-------------|
| `f1/circuits.csv` | Circuit info (name, location, country). |
| `f1/constructors.csv` | Team / constructor info. |
| `f1/constructor_results.csv` | Constructor results per race. |
| `f1/constructor_standings.csv` | Constructor standings per race. |
| `f1/drivers.csv` | Driver info (name, nationality, DOB). |
| `f1/driver_standings.csv` | Driver standings per race. |
| `f1/lap_times.csv` | Lap-by-lap times for every driver. |
| `f1/pit_stops.csv` | Pit stop times and durations. |
| `f1/qualifying.csv` | Qualifying session times. |
| `f1/races.csv` | Race info (date, round, season, circuit). |
| `f1/results.csv` | Final race results. |
| `f1/seasons.csv` | Season list with Wikipedia URL. |
| `f1/sprint_results.csv` | Sprint race results. |
| `f1/status.csv` | Race-finish status codes (Finished, DNF, etc.). |

`f1_clean.csv` is the **pre-processed merged dataset** produced by `f1_exploration.ipynb`. All modeling notebooks read from this file, not from the raw CSVs.

---

## 3. Notebooks

Run order: **start with `f1_exploration.ipynb`** to generate `f1_clean.csv`, then run any of the modeling notebooks.

### `f1_exploration.ipynb` — Data exploration & cleaning
- Loads raw CSVs from `f1/`, merges them, handles missing values, engineers features.
- Produces `f1_clean.csv` used by every other notebook.
- Outputs: `f1_heatmap.png`, `grid_vs_position.png`, `points_boxplot.png`, `top_drivers.png`.
- Run first.

| Feature correlations | Grid vs. finish position |
|---|---|
| ![Correlation heatmap](f1_heatmap.png) | ![Grid vs position](grid_vs_position.png) |

| Points distribution | Top drivers |
|---|---|
| ![Points boxplot](points_boxplot.png) | ![Top drivers](top_drivers.png) |

### Clustering (unsupervised) — `unsupervised/`

#### `unsupervised/kmeans.ipynb` — K-Means
- Standard K-Means on features `grid`, `laps`, `points`, `fastestLapSpeed`.
- Uses the elbow method and silhouette score to pick `k`.
- Outputs: `unsupervised/elbow_curve.png`, `unsupervised/kmeans_clusters.png`.

| Elbow curve | K-Means clusters |
|---|---|
| ![Elbow curve](unsupervised/elbow_curve.png) | ![K-Means clusters](unsupervised/kmeans_clusters.png) |

#### `unsupervised/kmeanspp.ipynb` — K-Means++
- Same task as K-Means, but with the K-Means++ initialisation and a comparison against random init (timing + ARI).
- Outputs: `unsupervised/kmeanspp_clusters.png`, `unsupervised/kmeanspp_boxplots.png`, `unsupervised/kmeanspp_comparison.png`.

| K-Means++ clusters | Init comparison |
|---|---|
| ![K-Means++ clusters](unsupervised/kmeanspp_clusters.png) | ![K-Means++ vs random init](unsupervised/kmeanspp_comparison.png) |

![K-Means++ feature boxplots](unsupervised/kmeanspp_boxplots.png)

#### `unsupervised/dbscan.ipynb` — DBSCAN
- Density-based clustering. Uses a k-distance plot to choose `eps` and the `2 × n_features` rule for `min_samples`.
- Outputs: `unsupervised/kdistance_plot.png`, `unsupervised/dbscan_result.png`.

| k-distance plot | DBSCAN result |
|---|---|
| ![k-distance plot](unsupervised/kdistance_plot.png) | ![DBSCAN clusters](unsupervised/dbscan_result.png) |

#### `unsupervised/hierarchical.ipynb` — Hierarchical (Agglomerative) clustering
- Compares linkage methods (ward / complete / average / single) with dendrograms and silhouette scores.
- Outputs: `unsupervised/dendrogram.png`, `unsupervised/hierarchical_clusters.png`.

| Dendrogram | Hierarchical clusters |
|---|---|
| ![Dendrogram](unsupervised/dendrogram.png) | ![Hierarchical clusters](unsupervised/hierarchical_clusters.png) |

### Classification (supervised — predict podium finish) — `supervised/`

All classification notebooks use the same train/test split (`random_state=42`, stratified, 20 % test) so results are directly comparable.

#### `supervised/knn.ipynb` — K-Nearest Neighbors
- Selects `k` via cross-validation, evaluates with accuracy, ROC-AUC, confusion matrix.
- Saves probabilities to `supervised/knn_y_prob.npy` and ground truth to `supervised/y_test.npy` for cross-model ROC comparison.
- Outputs: `supervised/knn_k_selection.png`, `supervised/knn_confusion.png`, `supervised/knn_roc.png`.

| k selection | Confusion matrix | ROC curve |
|---|---|---|
| ![KNN k selection](supervised/knn_k_selection.png) | ![KNN confusion](supervised/knn_confusion.png) | ![KNN ROC](supervised/knn_roc.png) |

#### `supervised/svm.ipynb` — Support Vector Machine
- `GridSearchCV` over `C` and `gamma` with an RBF kernel.
- Saves `supervised/svm_y_prob.npy`.
- Outputs: `supervised/svm_gridsearch.png`, `supervised/svm_confusion.png`, `supervised/svm_roc.png`.

| Grid search | Confusion matrix | ROC curve |
|---|---|---|
| ![SVM grid search](supervised/svm_gridsearch.png) | ![SVM confusion](supervised/svm_confusion.png) | ![SVM ROC](supervised/svm_roc.png) |

#### `supervised/decision_tree.ipynb` — Decision Tree
- `GridSearchCV` over `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion` (gini / entropy).
- Saves `supervised/dt_y_prob.npy`.
- Outputs: `supervised/dt_depth_search.png`, `supervised/dt_confusion.png`, `supervised/dt_roc.png`, `supervised/dt_importances.png`, `supervised/dt_tree.png`.

| Depth sweep | Confusion matrix | ROC curve |
|---|---|---|
| ![DT depth search](supervised/dt_depth_search.png) | ![DT confusion](supervised/dt_confusion.png) | ![DT ROC](supervised/dt_roc.png) |

| Feature importances | Tree (top 3 levels) |
|---|---|
| ![DT importances](supervised/dt_importances.png) | ![DT visualization](supervised/dt_tree.png) |

#### `supervised/linear_models.ipynb` — Logistic Regression & LDA
- Trains Logistic Regression (with L2 regularisation grid search) and Linear Discriminant Analysis.
- Loads the `.npy` files saved by `knn.ipynb`, `svm.ipynb`, and `decision_tree.ipynb` to plot a combined ROC across all five models.
- Outputs: `supervised/lr_coefficients.png`, `supervised/lr_confusion.png`, `supervised/roc_all_models.png`.

| LR coefficients | LR confusion matrix |
|---|---|
| ![LR coefficients](supervised/lr_coefficients.png) | ![LR confusion](supervised/lr_confusion.png) |

![Combined ROC across all models](supervised/roc_all_models.png)

---

## 4. Generated artifacts

These files are produced by running the notebooks — you do not need to commit or track them manually:

- **Intermediate data**: `f1_clean.csv` (root), `supervised/y_test.npy`, `supervised/knn_y_prob.npy`, `supervised/svm_y_prob.npy`, `supervised/dt_y_prob.npy`
- **Plots**: exploration plots (`f1_heatmap.png`, `grid_vs_position.png`, `points_boxplot.png`, `top_drivers.png`) in the project root; model plots inside `supervised/` and `unsupervised/`

---

## 5. How to run

1. Activate the venv: `source venv/bin/activate`
2. Start Jupyter: `jupyter notebook`
3. Open `f1_exploration.ipynb` and **Run All** cells to generate `f1_clean.csv`.
4. Open any modeling notebook and **Run All** cells.
5. For the cross-model ROC plot in `linear_models.ipynb`, run `knn.ipynb`, `svm.ipynb`, and `decision_tree.ipynb` first so their `.npy` outputs exist.

---

## 6. Troubleshooting

- **`FileNotFoundError: f1_clean.csv`** → run `f1_exploration.ipynb` first.
- **`FileNotFoundError: knn_y_prob.npy`, `svm_y_prob.npy`, or `dt_y_prob.npy`** → run `knn.ipynb`, `svm.ipynb`, and `decision_tree.ipynb` before the final ROC cell in `linear_models.ipynb`.
- **Module not found** → make sure the virtual environment is activated and dependencies are installed.
