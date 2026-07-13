# Phase 1 - Setup and Run Guide

## What this does
Trains ML models (XGBoost, 1D CNN, TCN) on DGS sign language videos to classify sentiment (negative, neutral, positive) and produces Sentiment Guidelines for Phase 2.

---

## Step 1 - Install VS Code and Python

1. Download VS Code: https://code.visualstudio.com/
2. Open VS Code, go to Extensions (left sidebar) and install:
   - "Python" by Microsoft
   - "Jupyter" by Microsoft

Download Python 3.11 from https://www.python.org/downloads/ if you do not have it. Make sure to check "Add Python to PATH" during installation.

---

## Step 2 - Set up the project environment

Open a terminal in VS Code and run these commands one by one:

```bash
cd phase1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This takes 3-5 minutes. You only need to do this once.

---

## Step 3 - Download the dataset

1. Go to: https://zenodo.org/records/18879038
2. Download all files from the Files section (not the Export section)
3. Place all CSV files directly inside the `data/` folder — not inside a subfolder

Your `data/` folder should contain:
```
data/
  DGS-Fabeln-1-SE-Labels.csv
  DGS-Fabeln-1-SE-MotionFeatures.csv
  DGS-Fabeln-1-SE-MediaPipe-1-DHUDI.csv
  DGS-Fabeln-1-SE-MediaPipe-2-FrauHolle.csv
  DGS-Fabeln-1-SE-MediaPipe-3-DerWolf.csv
  DGS-Fabeln-1-SE-MediaPipe-4-Schneewittchen.csv
  DGS-Fabeln-1-SE-MediaPipe-5-HaenselUndGretel.csv
  DGS-Fabeln-1-SE-MediaPipe-6-Dornroeschen.csv
  DGS-Fabeln-1-SE-MediaPipe-7-BremerStadtmusikanten.csv
```

---

## Step 4 - Run the notebooks in order

Navigate into the notebooks folder first:

```bash
cd phase1/notebooks
```

Then run each notebook in order:

```bash
../venv/bin/python 01_data_exploration.py
../venv/bin/python 02_feature_engineering.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../venv/bin/python 03_baseline_xgboost.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../venv/bin/python 04_temporal_cnn_tcn.py
../venv/bin/python 05_mapping_rules.py
```

Note: notebooks 03 and 04 require the OMP_NUM_THREADS flags on Mac to avoid a crash. Notebook 04 takes 30-60 minutes on CPU, which is normal.

Close any chart windows that pop up during notebook 02 to allow the script to continue.

---

## Step 5 - View your results

All outputs are saved in the `outputs/` folder:

- `outputs/plots/` - figures
- `outputs/rules/guidelines_FINAL.csv` - Sentiment Guidelines for Phase 2
- `outputs/rules/valence_scores_xgboost.csv` - Input Valence for Phase 2
- `outputs/model_comparison.csv` - model comparison table

---

## Actual results (body features only, no face)

| Model | Type | Balanced Accuracy |
|-------|------|------------------|
| XGBoost (paper baseline) | Classical ML, with face features | 0.631 |
| XGBoost (ours) | Classical ML, tabular features | 0.596 |
| 1D CNN | Neural network, temporal sequences | 0.589 |
| TCN | Advanced neural network, long-range patterns | 0.552 |

Face features were excluded because they are not implemented in Phase 2.

---

## Troubleshooting

**"No file matching Labels found"**
Make sure all CSV files are placed directly in `data/`, not inside a subfolder.

**"No module named src"**
You are running the notebook from the wrong folder. Make sure you are inside `phase1/notebooks/` before running.

**"Segmentation fault" on notebook 03 or 04**
Use the OMP_NUM_THREADS=1 prefix as shown above. This is a known Mac issue with XGBoost threading.

**Notebook 04 is very slow**
Normal without a GPU. 30-60 minutes on CPU is expected.
