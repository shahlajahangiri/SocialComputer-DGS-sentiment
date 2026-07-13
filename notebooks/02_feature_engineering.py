
# Notebook 2: Feature Selection
# Goal: From raw features select the most informative ones for sentiment classification.
# This notebook does SELECTION only (no new features generated from existing ones).

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from src.dataset import load_tabular, get_xy

df, feature_cols, label_col = load_tabular()
X, y, feature_names, scaler = get_xy(scale=False)   # raw, unscaled

print(f"Starting with {len(feature_names)} features")

# STEP 0: Remove face features (not implemented in our system)
# The paper notes face tracking as a known limitation.
# Keeping face features would inflate results we cannot reproduce in Phase 2.
FACE_KEYWORDS = ["eye", "mouth", "brow", "lip", "nose", "face", "cheek",
                 "jaw", "smile", "neutral", "funnel", "roll", "pucker"]
face_mask = [not any(kw in f.lower() for kw in FACE_KEYWORDS) for f in feature_names]
X = X[:, face_mask]
feature_names = [f for f, keep in zip(feature_names, face_mask) if keep]
n_removed = sum(not k for k in face_mask)
print(f"Removed {n_removed} face features → {len(feature_names)} body/motion features remain")

# STEP 1: Remove near-zero-variance features
# Logic: a feature that barely changes across all segments carries no information
# for distinguishing sentiments regardless of which model we use.
# This is a model-agnostic, computationally cheap filter that removes obvious noise
# before applying any supervised selection method.
selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X)
kept_mask = selector_var.get_support()
feature_names_var = [f for f, k in zip(feature_names, kept_mask) if k]
print(f"After variance threshold: {X_var.shape[1]} features (removed {len(feature_names) - X_var.shape[1]})")

# STEP 2: Mutual Information selection
# Mutual Information (MI) measures statistical dependency between a feature and the labels
# in any form — including non-linear patterns that a linear test like ANOVA would miss.
# We use MI as the sole supervised filter to avoid the cascading filter error:
# applying ANOVA first and then MI means MI only sees the features ANOVA already approved,
# which biases the selection toward ANOVA's linear assumptions and discards useful
# non-linear features before MI even gets a chance to evaluate them.
# By using MI alone, every body feature gets a fair chance to prove its relevance.
# We keep the top 100 features.
selector_mi = SelectKBest(mutual_info_classif, k=min(100, X_var.shape[1]))
X_misel = selector_mi.fit_transform(X_var, y)
mi_scores = selector_mi.scores_
kept_mi_mask = selector_mi.get_support()
feature_names_final = [f for f, k in zip(feature_names_var, kept_mi_mask) if k]
print(f"After Mutual Information selection (top 100): {X_misel.shape[1]} features")

# Plot top features by MI score
top_idx = np.argsort(mi_scores[kept_mi_mask])[::-1][:20]
top_feats = [feature_names_final[i] for i in top_idx]
top_scores = [mi_scores[kept_mi_mask][i] for i in top_idx]

plt.figure(figsize=(10, 5))
plt.barh(top_feats[::-1], top_scores[::-1], color="#3498db")
plt.xlabel("Mutual Information Score")
plt.title("Top 20 Features by Mutual Information")
plt.tight_layout()
plt.savefig("../outputs/plots/feature_mi_scores.png", dpi=150)
plt.show()

# STEP 3: Scaling (StandardScaler)
# Scaling transforms each feature so that it has mean = 0 and standard deviation = 1.
# For each feature value x: scaled_x = (x - mean) / std
# Without scaling, features with large numerical ranges (e.g. distance in pixels: 0-500)
# would dominate over features with small ranges (e.g. angles: 0-3),
# even if the smaller-range feature is more informative.
# This is especially critical for the CNN and TCN models which are sensitive to input scale.
# XGBoost is less sensitive to scaling but we apply it uniformly so all models
# receive identical input.
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X_misel)

# Build final dataframe and save
processed_df = pd.DataFrame(X_scaled, columns=feature_names_final)
processed_df["label"] = y
processed_df.to_csv("../outputs/processed_dataset.csv", index=False)

print(f"\nProcessed dataset saved: {X_scaled.shape}")
print(f"  → ../outputs/processed_dataset.csv")
print(f"\nLabel distribution in processed data:")
print(pd.Series(y).value_counts().rename({0:"negative", 1:"neutral", 2:"positive"}))
print("\nNext → run 03_baseline_xgboost.py")
