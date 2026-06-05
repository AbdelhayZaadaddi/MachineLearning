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

### Clustering (unsupervised)

#### `kmeans.ipynb` — K-Means
- Standard K-Means on features `grid`, `laps`, `points`, `fastestLapSpeed`.
- Uses the elbow method and silhouette score to pick `k`.
- Outputs: `elbow_curve.png`, `kmeans_clusters.png`.

#### `kmeanspp.ipynb` — K-Means++
- Same task as K-Means, but with the K-Means++ initialisation and a comparison against random init (timing + ARI).
- Outputs: `kmeanspp_clusters.png`, `kmeanspp_boxplots.png`, `kmeanspp_comparison.png`.

#### `dbscan.ipynb` — DBSCAN
- Density-based clustering. Uses a k-distance plot to choose `eps` and the `2 × n_features` rule for `min_samples`.
- Outputs: `kdistance_plot.png`, `dbscan_result.png`.

#### `hierarchical.ipynb` — Hierarchical (Agglomerative) clustering
- Compares linkage methods (ward / complete / average / single) with dendrograms and silhouette scores.
- Outputs: `dendrogram.png`, `hierarchical_clusters.png`.

### Classification (supervised — predict podium finish)

All classification notebooks use the same train/test split (`random_state=42`, stratified, 20 % test) so results are directly comparable.

#### `knn.ipynb` — K-Nearest Neighbors
- Selects `k` via cross-validation, evaluates with accuracy, ROC-AUC, confusion matrix.
- Saves probabilities to `knn_y_prob.npy` and ground truth to `y_test.npy` for cross-model ROC comparison.
- Outputs: `knn_k_selection.png`, `knn_confusion.png`, `knn_roc.png`.

#### `svm.ipynb` — Support Vector Machine
- `GridSearchCV` over `C` and `gamma` with an RBF kernel.
- Saves `svm_y_prob.npy`.
- Outputs: `svm_gridsearch.png`, `svm_confusion.png`, `svm_roc.png`.

#### `linear_models.ipynb` — Logistic Regression & LDA
- Trains Logistic Regression (with L2 regularisation grid search) and Linear Discriminant Analysis.
- Loads the `.npy` files saved by `knn.ipynb` and `svm.ipynb` to plot a combined ROC across all four models.
- Outputs: `lr_coefficients.png`, `lr_confusion.png`, `roc_all_models.png`.

---

## 4. Generated artifacts

These files are produced by running the notebooks — you do not need to commit or track them manually:

- **Intermediate data**: `f1_clean.csv`, `y_test.npy`, `knn_y_prob.npy`, `svm_y_prob.npy`
- **Plots**: all `*.png` files in the project root

---

## 5. How to run

1. Activate the venv: `source venv/bin/activate`
2. Start Jupyter: `jupyter notebook`
3. Open `f1_exploration.ipynb` and **Run All** cells to generate `f1_clean.csv`.
4. Open any modeling notebook and **Run All** cells.
5. For the cross-model ROC plot in `linear_models.ipynb`, run `knn.ipynb` and `svm.ipynb` first so their `.npy` outputs exist.

---

## 6. Troubleshooting

- **`FileNotFoundError: f1_clean.csv`** → run `f1_exploration.ipynb` first.
- **`FileNotFoundError: knn_y_prob.npy` or `svm_y_prob.npy`** → run `knn.ipynb` and `svm.ipynb` before the final ROC cell in `linear_models.ipynb`.
- **Module not found** → make sure the virtual environment is activated and dependencies are installed.
