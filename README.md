# How to Build a Social Computer — Phase 1

Sentiment classification from German Sign Language (DGS) videos using body movement features. This phase trains ML models to classify sign language segments as negative, neutral, or positive sentiment, and produces mapping rules for Phase 2.

## Dataset

Download all files from [https://zenodo.org/records/18879038](https://zenodo.org/records/18879038) and place them in `phase1/data/`. The folder should contain:

- `DGS-Fabeln-1-SE-Labels.csv`
- `DGS-Fabeln-1-SE-MotionFeatures.csv`
- `DGS-Fabeln-1-SE-MediaPipe-1-DHUDI.csv` through `MediaPipe-7-BremerStadtmusikanten.csv`

## Setup

```bash
cd phase1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the notebooks

Run from inside the `phase1/notebooks/` directory in order:

```bash
cd phase1/notebooks

../venv/bin/python 01_data_exploration.py
../venv/bin/python 02_feature_engineering.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../venv/bin/python 03_baseline_xgboost.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../venv/bin/python 04_temporal_cnn_tcn.py
../venv/bin/python 05_mapping_rules.py
```

Note: notebooks 03 and 04 require the thread limit flags on Mac to prevent a crash. Notebook 04 takes 30-60 minutes to complete.

## Results

| Model | Balanced Accuracy |
|-------|------------------|
| XGBoost (paper baseline, with face) | 0.631 |
| XGBoost (ours, body only) | 0.596 |
| 1D CNN (ours, body only) | 0.589 |
| TCN (ours, body only) | 0.552 |

Face features were intentionally excluded as they are not implemented in Phase 2. The paper identifies face tracking as a known limitation.

## Outputs

All outputs are saved in `phase1/outputs/`:

- `outputs/rules/mapping_rules_FINAL.csv` — mapping rules for Phase 2
- `outputs/rules/valence_scores_xgboost.csv` — per-segment valence scores for Phase 2
- `outputs/models/` — trained model files
- `outputs/plots/` — figures
