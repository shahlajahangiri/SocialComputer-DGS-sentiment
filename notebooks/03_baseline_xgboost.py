# Goal: Replicate and *beat* the paper's XGBoost result (0.631 balanced accuracy).
# Why this first? Fast to train (seconds, not hours) / Gives us feature importance → first version of Guidelines / Sets the performance bar for the CNN to beat

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.models import build_xgboost
from src.train import cv_xgboost
from src.evaluate import print_report, plot_confusion_matrix, extract_guidelines_xgboost, export_valence_scores

# Load processed dataset
proc = pd.read_csv("../outputs/processed_dataset.csv")
feature_cols = [c for c in proc.columns if c != "label"]
X = proc[feature_cols].values.astype(np.float32)
y = proc["label"].values.astype(int)

print(f"Dataset: {X.shape}  |  Classes: {np.bincount(y)}")

# Handle class imbalance with SMOTE (creates synthetic minority-class samples so the model doesn't just predict "neutral")
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"After SMOTE: {X_resampled.shape}  |  Classes: {np.bincount(y_resampled)}")

# ── Grid Search for best hyperparameters ──────────────────────────────────────
# We search over the most impactful XGBoost hyperparameters.
# learning_rate controls how much each tree corrects the previous ones.
# max_depth controls how complex each individual tree can be.
# n_estimators is the total number of trees.
# subsample and colsample_bytree add randomness to prevent overfitting.
# min_child_weight controls the minimum data needed to create a new tree branch.
print("\n── Grid Search for XGBoost Hyperparameters ──")
print("This may take 5-10 minutes. Finding the best hyperparameters automatically...")

param_grid = {
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [3, 5, 7],
    "n_estimators":     [200, 500],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "min_child_weight": [1, 3],
}

base_xgb = XGBClassifier(

    eval_metric="mlogloss",
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    n_jobs=1,
)

cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring=make_scorer(balanced_accuracy_score),
    cv=cv_inner,
    n_jobs=1,
    verbose=1,
    refit=True,
)

grid_search.fit(X_resampled, y_resampled)

print(f"\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV balanced accuracy (grid search): {grid_search.best_score_:.4f}")

best_params = grid_search.best_params_

# Cross-validation (same protocol as the paper: stratified k-fold) using best params
print("\n── 5-fold Cross Validation with best parameters ──")
model = XGBClassifier(

    eval_metric="mlogloss",
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    n_jobs=1,
    early_stopping_rounds=30,
    **best_params,
)
fold_scores = cv_xgboost(model, X_resampled, y_resampled)

# Train final model on ALL data (for guideline extraction + Phase 2 valence export)
print("\n── Training final model on full dataset ──")
final_model = XGBClassifier(

    eval_metric="mlogloss",
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    n_jobs=1,
    early_stopping_rounds=30,
    **best_params,
)
final_model.fit(
    X_resampled, y_resampled,
    eval_set=[(X, y)],
    verbose=False,
)

# Evaluate on original (non-resampled) data
preds = final_model.predict(X)
print_report(y, preds, "XGBoost")
plot_confusion_matrix(y, preds, "XGBoost")

# Extract Guidelines (SHAP)
print("\n── Extracting Guidelines via SHAP ──")
guidelines_df = extract_guidelines_xgboost(final_model, X, feature_cols, top_k=30)

print("\nTop 10 Guidelines:")
print(guidelines_df.head(10)[["rank","feature_name","dominant_sentiment","global_importance"]].to_string(index=False))

# Export valence scores for Phase 2
segment_ids = list(range(len(X)))   # replace with actual segment IDs if available
valence_df = export_valence_scores(final_model, X, segment_ids, model_name="xgboost")

print("\nSample valence scores (first 5 rows):")
print(valence_df.head())

# Save the trained model
import joblib
joblib.dump(final_model, "../outputs/models/xgboost_final.pkl")
print("\n✓ Model saved → ../outputs/models/xgboost_final.pkl")

# Save best params so other notebooks can reference them
pd.DataFrame([best_params]).to_csv("../outputs/xgboost_best_params.csv", index=False)
print("✓ Best params saved → ../outputs/xgboost_best_params.csv")

print("\nNext → run 04_temporal_cnn_tcn.py  (the novel contribution!)")
