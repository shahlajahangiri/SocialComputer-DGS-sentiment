# Phase 1 — Code Documentation

This document explains what each file in the codebase does, how they connect, and what each function is responsible for. For setup and running instructions, see [SETUP_AND_RUN.md](SETUP_AND_RUN.md).

---

## Project Structure

```
phase1/
├── data/                  # raw dataset (not tracked in git)
├── notebooks/             # pipeline scripts, run in order 01 → 05
├── src/                   # shared logic imported by the notebooks
│   ├── dataset.py         # loading and preprocessing the DGS-Fabeln-1-SE data
│   ├── models.py          # XGBoost, 1D CNN, and TCN model definitions
│   ├── train.py           # training loops and cross-validation
│   └── evaluate.py        # metrics, plots, SHAP-based sentiment guidelines
├── outputs/                # all generated results
│   ├── plots/              # figures
│   ├── models/              # trained model files
│   └── rules/                # sentiment guidelines, valence scores, JSON output
├── README.md
├── SETUP_AND_RUN.md
└── requirements.txt
```

---

## Notebooks (run in order)

| # | File | What it does |
|---|------|---------------|
| 01 | `01_data_exploration.py` | Loads the labels file, prints class distribution, explores the raw dataset |
| 02 | `02_feature_engineering.py` | Removes face features, applies Variance Threshold + Mutual Information selection, scales features, saves `processed_dataset.csv` |
| 03 | `03_baseline_xgboost.py` | Runs grid search for XGBoost hyperparameters, trains the final model, extracts SHAP-based guidelines, exports valence scores |
| 04 | `04_temporal_cnn_tcn.py` | Trains 1D CNN and TCN directly on raw MediaPipe sequences, compares learning rates, saves training curve plots |
| 05 | `05_mapping_rules.py` | Produces the final `guidelines_FINAL.csv` and `phase2_guidelines.json` — the two outputs handed off to Phase 2 |

Each notebook must be run from inside the `notebooks/` folder (`cd phase1/notebooks`) because of the relative import `sys.path.insert(0, "..")`.

---

## `src/dataset.py`

Handles all data loading. Two separate data representations are produced, one for XGBoost and one for the neural network models.

### `load_tabular(data_dir=DATA_DIR)`
Loads the Labels file and the MotionFeatures file, merges them on a `Story` + `id` composite key (since segment IDs repeat across different fairy tales), drops rows with ambiguous ("multi") labels, and returns a merged DataFrame along with the list of feature column names.

### `get_xy(data_dir=DATA_DIR, scale=True)`
Calls `load_tabular()` and converts the result into NumPy arrays `X` (features) and `y` (integer-encoded labels: 0=negative, 1=neutral, 2=positive). Missing values are replaced with the column median. If `scale=True`, applies `StandardScaler`.

### `load_temporal(data_dir=DATA_DIR, max_len=300)`
Loads all seven MediaPipe CSV files (frame-by-frame body tracking, one file per fairy tale), groups rows by segment, and pads or truncates every segment to `max_len` frames. Returns `sequences` of shape `(N, max_len, n_features)`, along with the corresponding `labels` and segment IDs. This is the data format used by the CNN and TCN models in notebook 04.

### `TabularDataset` / `TemporalDataset`
PyTorch `Dataset` wrappers. `TemporalDataset` transposes the sequence array from `(N, time, features)` to `(N, features, time)`, since PyTorch's `Conv1d` expects the channel dimension (features) before the time dimension.

### Helper functions
`_find_file`, `_find_label_col`, `_find_id_col`, `_find_col_containing` — these make the loader resilient to small naming differences in the downloaded CSV files (e.g. column names varying slightly between dataset versions).

---

## `src/models.py`

Defines all three models used in Phase 1.

### `build_xgboost(n_classes=3, **kwargs)`
Returns an `XGBClassifier` configured for 3-class classification. Default hyperparameters are provided, but any can be overridden via `**kwargs` — this is how notebook 03's grid search passes in the best-found hyperparameters.

### `CNN1D(in_channels, n_classes=3, dropout=0.4)`
A 1D convolutional network with three convolutional blocks (Conv → BatchNorm → ReLU → MaxPool → Dropout), followed by global average pooling and a fully connected classifier. Operates directly on raw per-frame MediaPipe sequences.

### `TCN(in_channels, n_classes=3, n_layers=4, hidden_channels=64, kernel_size=3, dropout=0.3)`
A Temporal Convolutional Network using dilated causal convolutions (dilation doubles at each layer: 1, 2, 4, 8...), allowing the model to see an exponentially larger time window without adding depth. Built from stacked `_TCNBlock` residual blocks.

---

## `src/train.py`

Contains the training loops and cross-validation logic shared by all models.

### `cv_xgboost(model, X, y, n_splits=5, random_state=42)`
Runs 5-fold stratified cross-validation for XGBoost, printing the balanced accuracy per fold and the overall mean ± standard deviation.

### `train_epoch(model, loader, optimizer, criterion)` / `eval_epoch(model, loader, criterion)`
One training/evaluation pass over a PyTorch `DataLoader`. Used by both the CNN and TCN.

### `cv_pytorch(model_fn, dataset, y, n_splits=5, epochs=50, batch_size=16, lr=1e-3, random_state=42)`
Runs 5-fold stratified cross-validation for any PyTorch model. `model_fn` is a callable that returns a fresh model instance for each fold (so no weights leak between folds). Uses class-weighted `CrossEntropyLoss` to address the imbalanced classes, and a `CosineAnnealingLR` scheduler.

`DEVICE` is set automatically to `cuda` if available, otherwise `cpu`.

---

## `src/evaluate.py`

Contains everything related to turning a trained model into interpretable output.

### `print_report(y_true, y_pred, model_name="Model")` / `plot_confusion_matrix(...)`
Standard classification report and confusion matrix, saved to `outputs/plots/`.

### `extract_guidelines_xgboost(model, X, feature_names, top_k=20, save=True)`
Runs SHAP's `TreeExplainer` on the trained XGBoost model to compute per-feature, per-class importance scores. Handles both the old SHAP output format (a list of arrays, one per class) and the newer format (a single 3D array), since SHAP's API changed between versions. Returns a ranked DataFrame of the top `k` features with their SHAP importance per sentiment class, and saves it as `guidelines_xgboost.csv`.

### `plot_shap_summary(model, X, feature_names, save=True)`
Generates SHAP beeswarm plots showing feature impact per class.

### `export_valence_scores(model, X, segment_ids, model_name="xgboost", save=True)`
Converts the model's predicted class probabilities into a single continuous valence score per segment, using:
```
valence = prob_negative × (-1) + prob_neutral × 0 + prob_positive × (+1)
```
Saves the result as `valence_scores_xgboost.csv` — this is the "Input Valence" file used directly by Phase 2.

---

## Data Flow Summary

```
data/*.csv
   │
   ├─→ load_tabular() / get_xy()  ──→  processed_dataset.csv  (notebook 02)
   │                                        │
   │                                        ├─→ build_xgboost() + cv_xgboost()  (notebook 03)
   │                                        │        │
   │                                        │        ├─→ extract_guidelines_xgboost()  → guidelines_xgboost.csv
   │                                        │        └─→ export_valence_scores()       → valence_scores_xgboost.csv
   │                                        │
   └─→ load_temporal()  ──→  CNN1D / TCN + cv_pytorch()  (notebook 04)
                                                 │
                                                 └─→ model_comparison.csv

notebook 05: combines guidelines_xgboost.csv + valence_scores_xgboost.csv
             → maps features to MMS parameters
             → guidelines_FINAL.csv
             → phase2_guidelines.json   (final Phase 2 handoff)
```

---

## Notes on Known Limitations

- Dataset size (517 segments) limits neural network performance relative to XGBoost (see `README.md` for the full results comparison).
- Face features are intentionally excluded throughout, since the MMS Player does not support facial modulation.
- SHAP importance scores (`guidelines_FINAL.csv`) are used as **evidence** to inform Phase 2's manually assigned modulation values (they are not used as direct modulation weights).
