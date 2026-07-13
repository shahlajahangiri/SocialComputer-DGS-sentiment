# Notebook 5 - Sentiment Guidelines + Phase 2 Handoff
# Goal: Generate the "Sentiment Guidelines" output from the Phase 1 workflow.
# Guidelines = a structured table + JSON that tells Phase 2:
# "for each valence (negative/neutral/positive), which MMS parameters are most important,
#  and what alpha weight should be applied in the modulator equation?"

import sys
sys.path.insert(0, "..")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

# Load the processed dataset + trained XGBoost model
proc = pd.read_csv("../outputs/processed_dataset.csv")
feature_cols = [c for c in proc.columns if c != "label"]
X = proc[feature_cols].values.astype(np.float32)
y = proc["label"].values.astype(int)

xgb_model = joblib.load("../outputs/models/xgboost_final.pkl")
print("Models loaded.")

# SHAP-based Sentiment Guidelines (already computed in notebook 3)
from src.evaluate import extract_guidelines_xgboost, plot_shap_summary

guidelines_df = extract_guidelines_xgboost(xgb_model, X, feature_cols, top_k=30)
print("\nFull Sentiment Guidelines table:")
print(guidelines_df.to_string(index=False))

# ── MMS Parameter Mapping ──────────────────────────────────────────────────────
# The MMS (Multimodal Sign language player) inflects animation using these parameters:
#   domhandreloc, ndomhandreloc, domshoulderreloc, ndomshoulderreloc,
#   torsoreloc, torsorot, headrot, speed (arousal-related)
#
# We map each feature from our dataset to the closest MMS parameter.
# The alpha value in the modulator equation x_new = clip(x + m * alpha * v * V_max, L, U)
# comes from the SHAP importance score of the corresponding feature for each valence.
# A higher alpha means Phase 2 should apply a stronger modulation for that parameter.

MMS_KEYWORD_MAP = {
    "domhandreloc":     ["right_hand", "r_wrist", "right_wrist", "right_hand_wrist",
                         "right_index", "right_thumb", "right_pinky"],
    "ndomhandreloc":    ["left_hand", "l_wrist", "left_wrist", "left_hand_wrist",
                         "left_index", "left_thumb", "left_pinky"],
    "domshoulderreloc": ["r_shoulder", "right_shoulder", "right_arm_angle"],
    "ndomshoulderreloc":["l_shoulder", "left_shoulder", "left_arm_angle"],
    "torsoreloc":       ["hip"],
    "torsorot":         ["torso_pitch", "torso_yaw", "torso"],
    "headrot":          ["head_pitch", "head_yaw", "head_roll", "head"],
    "speed":            ["accum_dist", "peaks_per_s", "velocity", "speed"],
}

SENTIMENTS = ["negative", "neutral", "positive"]
SHAP_COLS  = {"negative": "shap_negative", "neutral": "shap_neutral", "positive": "shap_positive"}


def match_mms_param(feature_name):
    feat_lower = feature_name.lower()
    for mms_param, keywords in MMS_KEYWORD_MAP.items():
        if any(kw in feat_lower for kw in keywords):
            return mms_param
    return None


guidelines_df["mms_param"] = guidelines_df["feature_name"].apply(match_mms_param)

# For each valence and each MMS parameter, compute alpha = mean SHAP importance
# of all features that map to that MMS parameter, then normalize so max alpha = 1.0
valence_guidelines = {}

for sentiment in SENTIMENTS:
    shap_col = SHAP_COLS[sentiment]
    param_scores = {}

    for mms_param in MMS_KEYWORD_MAP:
        subset = guidelines_df[guidelines_df["mms_param"] == mms_param]
        if len(subset) > 0:
            param_scores[mms_param] = float(subset[shap_col].mean())
        else:
            param_scores[mms_param] = 0.0

    # Normalize: divide by max so alpha values are in [0, 1]
    max_score = max(param_scores.values()) if max(param_scores.values()) > 0 else 1.0
    valence_guidelines[sentiment] = {
        param: round(score / max_score, 4)
        for param, score in param_scores.items()
    }

# ── Make alpha values relative to neutral (neutral = 0.0) ─────────────────────
# Subtract the neutral alpha from each valence so that:
#   neutral  → 0.0  (baseline, no modulation)
#   positive → positive value means more movement than neutral
#   negative → negative value means less/different movement than neutral
neutral_alphas = valence_guidelines["neutral"]
relative_guidelines = {}
for sentiment in SENTIMENTS:
    relative_guidelines[sentiment] = {
        param: round(valence_guidelines[sentiment].get(param, 0) - neutral_alphas.get(param, 0), 4)
        for param in MMS_KEYWORD_MAP
    }
# neutral should now be all zeros
relative_guidelines["neutral"] = {param: 0.0 for param in MMS_KEYWORD_MAP}

# ── Build JSON output ──────────────────────────────────────────────────────────
# Structure: for each valence → each MMS parameter → alpha value
# Phase 2 uses: x_new = clip(x + m * alpha * v * V_max, L, U)
# where alpha comes from here, V_max comes from Table 5.1 of Mishra thesis (from teammates),
# v is the continuous valence score from valence_scores_xgboost.csv,
# and m is the modulation direction (+1 or -1).

phase2_json = {
    "_description": (
        "Phase 1 Sentiment Guidelines. "
        "Alpha values are relative to neutral (neutral = 0.0). "
        "Positive alpha = more movement than neutral; negative alpha = less movement than neutral. "
        "Use in: x_new = clip(x + m * alpha * v * V_max, L, U). "
        "V_max values must be provided by teammates from Table 5.1 of Mishra thesis."
    ),
    "valence_guidelines": relative_guidelines,
}

json_path = "../outputs/rules/phase2_guidelines.json"
with open(json_path, "w") as f:
    json.dump(phase2_json, f, indent=2)

print(f"\nPhase 2 JSON guidelines saved → {json_path}")
print("\nJSON preview:")
print(json.dumps(phase2_json, indent=2))

# ── Plot: which body part drives each sentiment? ───────────────────────────────
def _extract_body_part(feat_name):
    feat = feat_name.lower()
    for part in ["shoulder", "elbow", "wrist", "hip", "hand",
                 "head", "spine", "torso", "nose", "neck"]:
        if part in feat:
            return part
    return "other"

guidelines_df["body_part"] = guidelines_df["feature_name"].apply(_extract_body_part)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, sentiment, col in zip(axes, SENTIMENTS, SHAP_COLS.values()):
    top10 = guidelines_df.nlargest(10, col)
    ax.barh(top10["feature_name"].str[:30].values[::-1],
            top10[col].values[::-1],
            color={"negative":"#e74c3c","neutral":"#95a5a6","positive":"#2ecc71"}[sentiment])
    ax.set_title(f"Top features - '{sentiment}'")
    ax.set_xlabel("SHAP importance")

plt.suptitle("Sentiment Guidelines: Feature - Sentiment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/plots/guidelines_by_sentiment.png", dpi=150)
plt.show()

# ── Plot: alpha values per MMS parameter per valence ──────────────────────────
mms_params = list(MMS_KEYWORD_MAP.keys())
x = np.arange(len(mms_params))
width = 0.25
colors = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}

fig, ax = plt.subplots(figsize=(12, 5))
for i, sentiment in enumerate(SENTIMENTS):
    alphas = [relative_guidelines[sentiment].get(p, 0) for p in mms_params]
    ax.bar(x + i * width, alphas, width, label=sentiment, color=colors[sentiment])

ax.set_xticks(x + width)
ax.set_xticklabels(mms_params, rotation=30, ha="right")
ax.set_ylabel("Alpha (normalized SHAP importance)")
ax.set_title("MMS Parameter Alpha Values per Valence (relative to neutral=0)\n(used in Phase 2 modulator equation)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/plots/mms_alpha_per_valence.png", dpi=150)
plt.show()

# ── Generate the final Sentiment Guidelines CSV for Phase 2 ──────────────────
def make_guideline_text(row):
    feat = row["feature_name"]
    sent = row["dominant_sentiment"]
    imp  = row["global_importance"]
    return f"High '{feat}' indicates sentiment: {sent}  (importance={imp:.3f})"

guidelines_df["guideline_description"] = guidelines_df.apply(make_guideline_text, axis=1)

final_guidelines = guidelines_df[["rank", "feature_name", "body_part", "mms_param",
                                   "dominant_sentiment", "global_importance",
                                   "shap_negative", "shap_neutral", "shap_positive",
                                   "guideline_description"]]

final_guidelines.to_csv("../outputs/rules/guidelines_FINAL.csv", index=False)
print("\nFinal Sentiment Guidelines saved:")
print("  → ../outputs/rules/guidelines_FINAL.csv")

# Also export valence scores per segment (continuous -1 to +1)
valence_df = pd.read_csv("../outputs/rules/valence_scores_xgboost.csv")
print("\nSample valence scores for Phase 2:")
print(valence_df.head(10))

# Summary
print("\n" + "="*60)
print("  PHASE 1 COMPLETE - SUMMARY")
print("="*60)

comparison = pd.read_csv("../outputs/model_comparison.csv") \
    if os.path.exists("../outputs/model_comparison.csv") else pd.DataFrame()
if not comparison.empty:
    print(comparison.to_string(index=False))

print(f"""
Outputs produced:
  outputs/processed_dataset.csv             - selected features (body only)
  outputs/model_comparison.csv              - all model scores
  outputs/models/xgboost_final.pkl          - trained XGBoost
  outputs/models/cnn1d_final.pt             - trained 1D CNN
  outputs/models/tcn_final.pt               - trained TCN
  outputs/rules/guidelines_FINAL.csv        - SENTIMENT GUIDELINES for Phase 2
  outputs/rules/phase2_guidelines.json      - JSON guidelines (alpha per MMS param per valence)
  outputs/rules/valence_scores_xgboost.csv  - per-segment valence scores for Phase 2
  outputs/plots/                            - all figures including training curves

PHASE 2 inputs:
  - phase2_guidelines.json         - alpha values per MMS parameter per valence
  - valence_scores_xgboost.csv     - Input Valence (v) for each MMS segment
  - V_max values needed from teammates (Table 5.1 of Mishra thesis)

Modulator equation:
  x_new = clip(x + m * alpha * v * V_max, L, U)
  where alpha comes from phase2_guidelines.json
""")
