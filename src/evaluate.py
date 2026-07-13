"""
Evaluation metrics, plots, and — most importantly — Sentiment Guidelines extraction.

Sentiment Guidelines are the Phase 1 output that feeds into Phase 2 as "Input Valence."
They tell Phase 2: "when these motion features have these values → apply this sentiment."
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
LABEL_NAMES = ["negative", "neutral", "positive"]


# ──────────────────────────────────────────────────────────────
# Classification report + confusion matrix
# ──────────────────────────────────────────────────────────────

def print_report(y_true, y_pred, model_name="Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    ba = balanced_accuracy_score(y_true, y_pred)
    print(f"  Balanced Accuracy: {ba:.4f}  (paper baseline: 0.631)")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save=True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "plots", f"cm_{model_name.lower().replace(' ','_')}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# SHAP-based Sentiment Guidelines (XGBoost)
# ──────────────────────────────────────────────────────────────

def extract_guidelines_xgboost(model, X, feature_names, top_k=20, save=True):
    """
    Use SHAP values to extract interpretable sentiment guidelines from XGBoost.

    Output CSV columns:
      feature_name | importance | negative_shap | neutral_shap | positive_shap
                   | dominant_sentiment

    This CSV is the "Sentiment Guidelines" block in the Phase 1 workflow.
    It tells Phase 2: which features signal which sentiment.
    """
    print("Computing SHAP values (may take ~30 seconds) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle both old SHAP format (list of arrays) and new format (3D array)
    # Old: list of 3 arrays each shape (n_samples, n_features)
    # New: single array shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        mean_shap = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])  # (3, n_features)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0).T  # → (n_classes, n_features)
    global_importance = mean_shap.mean(axis=0)

    top_idx = np.argsort(global_importance)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_idx]

    rules = []
    for i, feat_idx in enumerate(top_idx):
        feat = feature_names[feat_idx]
        neg_s  = mean_shap[0, feat_idx]
        neu_s  = mean_shap[1, feat_idx]
        pos_s  = mean_shap[2, feat_idx]
        dominant = LABEL_NAMES[np.argmax([neg_s, neu_s, pos_s])]
        rules.append({
            "rank": i + 1,
            "feature_name": feat,
            "global_importance": global_importance[feat_idx],
            "shap_negative": neg_s,
            "shap_neutral": neu_s,
            "shap_positive": pos_s,
            "dominant_sentiment": dominant,
        })

    rules_df = pd.DataFrame(rules)

    if save:
        path = os.path.join(OUTPUT_DIR, "rules", "guidelines_xgboost.csv")
        rules_df.to_csv(path, index=False)
        print(f"Guidelines saved → {path}")

    return rules_df


def plot_shap_summary(model, X, feature_names, save=True):
    """SHAP beeswarm plot — shows which features push toward each sentiment."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    for class_idx, class_name in enumerate(LABEL_NAMES):
        shap.summary_plot(
            shap_values[class_idx], X,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP Feature Impact — '{class_name}' sentiment")
        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, "plots", f"shap_{class_name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()


# ──────────────────────────────────────────────────────────────
# Valence score export (for Phase 2 input)
# ──────────────────────────────────────────────────────────────

def export_valence_scores(model, X, segment_ids, model_name="xgboost", save=True):
    """
    Export per-segment valence scores that Phase 2 will use as "Input Valence."

    Columns: segment_id | pred_label | prob_negative | prob_neutral | prob_positive | valence_score
    valence_score is a continuous value in [-1, +1]:
       -1 = strongly negative, 0 = neutral, +1 = strongly positive
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)          # sklearn / XGBoost
    else:
        raise ValueError("Model must have predict_proba. For PyTorch use softmax(logits).")

    preds = probs.argmax(axis=1)
    # continuous valence: weighted sum of class probabilities mapped to [-1, 0, +1]
    valence = probs[:, 0] * (-1) + probs[:, 1] * 0 + probs[:, 2] * 1

    df = pd.DataFrame({
        "segment_id"   : segment_ids,
        "pred_label"   : [LABEL_NAMES[p] for p in preds],
        "prob_negative": probs[:, 0].round(4),
        "prob_neutral" : probs[:, 1].round(4),
        "prob_positive": probs[:, 2].round(4),
        "valence_score": valence.round(4),
    })

    if save:
        path = os.path.join(OUTPUT_DIR, "rules", f"valence_scores_{model_name}.csv")
        df.to_csv(path, index=False)
        print(f"Valence scores saved → {path}")
        print("  → This file is the 'Input Valence' for Phase 2 (MMS emotional layer).")

    return df
