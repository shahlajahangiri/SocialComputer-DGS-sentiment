# Goal: Beat XGBoost by learning directly from raw per-frame MediaPipe sequences.
# Why this is the key innovation vs. the paper:
# Paper uses 396 *summary statistics* per segment (mean, std, etc.) = loses all temporal structure of signing
# We feed the raw frame-by-frame data into a CNN/TCN = the model *learns* which temporal patterns = which sentiment
#
# Requires: The raw MediaPipe CSVs from zenodo.org/records/18879038 (the 7 per-fairy-tale files)

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.dataset import load_temporal, TemporalDataset
from src.models import CNN1D, TCN
from src.train import cv_pytorch, train_epoch, eval_epoch, DEVICE
from src.evaluate import print_report, plot_confusion_matrix, export_valence_scores

# Load raw temporal sequences
# Each segment becomes a matrix: (n_frames, n_mediapipe_features)
# We pad/truncate all segments to max_len=300 frames (= ~10 sec at 30fps)
print("Loading raw MediaPipe sequences ...")
sequences, labels, seg_ids = load_temporal(max_len=300)

N, T, C = sequences.shape
print(f"\nSequences shape: {sequences.shape}")
print(f"  N = {N} segments")
print(f"  T = {T} frames (padded/truncated)")
print(f"  C = {C} features per frame")

# Create PyTorch Dataset
dataset = TemporalDataset(sequences, labels)

# ── Learning Rate Search ──────────────────────────────────────────────────────
# We try three learning rate values: fast (1e-2), default (1e-3), slow (1e-4).
# A learning rate that is too high causes the model to overshoot the best solution.
# A learning rate that is too low makes training very slow and may get stuck.
# We run a quick 3-fold CV for each LR and pick the best one before full training.
LR_CANDIDATES = [1e-2, 1e-3, 1e-4]

def make_cnn(in_channels):
    return CNN1D(in_channels=in_channels, n_classes=3, dropout=0.4)

def make_tcn(in_channels):
    return TCN(
        in_channels=in_channels,
        n_classes=3,
        n_layers=6,
        hidden_channels=64,
        kernel_size=3,
        dropout=0.3,
    )

print("\n" + "="*60)
print("  LEARNING RATE SEARCH (3-fold CV, 30 epochs each)")
print("="*60)

lr_results = {"CNN": {}, "TCN": {}}

for lr in LR_CANDIDATES:
    print(f"\n── CNN  lr={lr} ──")
    scores = cv_pytorch(
        model_fn=lambda: make_cnn(C),
        dataset=dataset,
        y=labels,
        n_splits=3,
        epochs=30,
        batch_size=16,
        lr=lr,
    )
    lr_results["CNN"][lr] = np.mean(scores)
    print(f"  CNN  lr={lr}  →  mean BA = {np.mean(scores):.4f}")

for lr in LR_CANDIDATES:
    print(f"\n── TCN  lr={lr} ──")
    scores = cv_pytorch(
        model_fn=lambda: make_tcn(C),
        dataset=dataset,
        y=labels,
        n_splits=3,
        epochs=30,
        batch_size=16,
        lr=lr,
    )
    lr_results["TCN"][lr] = np.mean(scores)
    print(f"  TCN  lr={lr}  →  mean BA = {np.mean(scores):.4f}")

# Pick best LR for each model
best_lr_cnn = max(lr_results["CNN"], key=lr_results["CNN"].get)
best_lr_tcn = max(lr_results["TCN"], key=lr_results["TCN"].get)
print(f"\nBest LR for CNN: {best_lr_cnn}  (BA={lr_results['CNN'][best_lr_cnn]:.4f})")
print(f"Best LR for TCN: {best_lr_tcn}  (BA={lr_results['TCN'][best_lr_tcn]:.4f})")

# Plot LR comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, model_name in zip(axes, ["CNN", "TCN"]):
    lrs = [str(lr) for lr in LR_CANDIDATES]
    bas  = [lr_results[model_name][lr] for lr in LR_CANDIDATES]
    ax.bar(lrs, bas, color="#3498db")
    ax.set_title(f"{model_name}: Balanced Accuracy by Learning Rate")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Mean Balanced Accuracy (3-fold)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(bas):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("../outputs/plots/lr_comparison.png", dpi=150)
plt.show()

# ════════════════════════════════════════════════════
# MODEL A: 1D CNN  (full 5-fold CV + training curves)
# ════════════════════════════════════════════════════

print("\n" + "═"*50)
print("MODEL A: 1D CNN")
print("═"*50)
print(f"Using best learning rate: {best_lr_cnn}")

cnn_demo = make_cnn(C).to(DEVICE)
demo_input = torch.randn(2, C, T).to(DEVICE)
demo_out = cnn_demo(demo_input)
print(f"Input:  {demo_input.shape}")
print(f"Output: {demo_out.shape}  (batch × 3 classes)")
print(f"Parameters: {sum(p.numel() for p in cnn_demo.parameters()):,}")

print("\nStarting CNN 5-fold cross-validation ...")
cnn_scores = cv_pytorch(
    model_fn=lambda: make_cnn(C),
    dataset=dataset,
    y=labels,
    n_splits=5,
    epochs=60,
    batch_size=16,
    lr=best_lr_cnn,
)

# Train final CNN with OneCycleLR scheduler + training curves
print("\nTraining final CNN model (with OneCycleLR scheduler) ...")
final_cnn = make_cnn(C).to(DEVICE)

class_counts = np.bincount(labels)
weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

full_loader = DataLoader(dataset, batch_size=16, shuffle=True)

EPOCHS_CNN = 80
optimizer_cnn = torch.optim.AdamW(final_cnn.parameters(), lr=best_lr_cnn, weight_decay=1e-4)
# OneCycleLR: ramps up the LR at the start then decays it — helps escape flat regions fast
scheduler_cnn = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_cnn, max_lr=best_lr_cnn * 10,
    steps_per_epoch=len(full_loader), epochs=EPOCHS_CNN
)

cnn_train_ba_history = []
for epoch in range(EPOCHS_CNN):
    loss, ba = train_epoch(final_cnn, full_loader, optimizer_cnn, criterion)
    scheduler_cnn.step()
    cnn_train_ba_history.append(ba)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | loss={loss:.4f} | train BA={ba:.4f}")

torch.save(final_cnn.state_dict(), "../outputs/models/cnn1d_final.pt")
print("✓ CNN model saved.")

# Training curve plot for CNN
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS_CNN + 1), cnn_train_ba_history, color="#3498db", label="Train BA")
plt.xlabel("Epoch")
plt.ylabel("Balanced Accuracy")
plt.title("CNN Training Curve (Balanced Accuracy per Epoch)")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/plots/cnn_training_curve.png", dpi=150)
plt.show()

# ════════════════════════════════════════════════════
# MODEL B: TCN  (full 5-fold CV + training curves)
# ════════════════════════════════════════════════════

print("\n" + "═"*50)
print("MODEL B: TCN  (expected to outperform CNN)")
print("═"*50)
print(f"Using best learning rate: {best_lr_tcn}")

tcn_demo = make_tcn(C).to(DEVICE)
out = tcn_demo(demo_input)
print(f"TCN output shape: {out.shape}  |  Parameters: {sum(p.numel() for p in tcn_demo.parameters()):,}")

print("\nStarting TCN 5-fold cross-validation ...")
tcn_scores = cv_pytorch(
    model_fn=lambda: make_tcn(C),
    dataset=dataset,
    y=labels,
    n_splits=5,
    epochs=60,
    batch_size=16,
    lr=best_lr_tcn,
)

# Train final TCN with ReduceLROnPlateau scheduler + training curves
# ReduceLROnPlateau: automatically halves the LR when the model stops improving
print("\nTraining final TCN model (with ReduceLROnPlateau scheduler) ...")
final_tcn = make_tcn(C).to(DEVICE)

EPOCHS_TCN = 80
optimizer_tcn = torch.optim.AdamW(final_tcn.parameters(), lr=best_lr_tcn, weight_decay=1e-4)
scheduler_tcn = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_tcn, mode="max", factor=0.5, patience=10
)

tcn_train_ba_history = []
for epoch in range(EPOCHS_TCN):
    loss, ba = train_epoch(final_tcn, full_loader, optimizer_tcn, criterion)
    scheduler_tcn.step(ba)   # ReduceLROnPlateau needs the metric to decide
    tcn_train_ba_history.append(ba)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | loss={loss:.4f} | train BA={ba:.4f}")

torch.save(final_tcn.state_dict(), "../outputs/models/tcn_final.pt")
print("✓ TCN model saved.")

# Training curve plot for TCN
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS_TCN + 1), tcn_train_ba_history, color="#e74c3c", label="Train BA")
plt.xlabel("Epoch")
plt.ylabel("Balanced Accuracy")
plt.title("TCN Training Curve (Balanced Accuracy per Epoch)")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/plots/tcn_training_curve.png", dpi=150)
plt.show()

# Combined training curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS_CNN + 1), cnn_train_ba_history, color="#3498db", label="CNN Train BA")
plt.plot(range(1, EPOCHS_TCN + 1), tcn_train_ba_history, color="#e74c3c", label="TCN Train BA")
plt.xlabel("Epoch")
plt.ylabel("Balanced Accuracy")
plt.title("CNN vs TCN Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/plots/training_curves_comparison.png", dpi=150)
plt.show()

# Compare all models
comparison = pd.DataFrame({
    "Model": ["XGBoost (paper)", "XGBoost (ours)", "1D CNN", "TCN"],
    "Balanced Accuracy": [
        0.631,
        0.593,
        np.mean(cnn_scores),
        np.mean(tcn_scores),
    ],
    "Type": ["Baseline (paper)", "Baseline (ours)", "Novel", "Novel"],
})
print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
print(comparison.to_string(index=False))
comparison.to_csv("../outputs/model_comparison.csv", index=False)
print("\n✓ Saved: ../outputs/model_comparison.csv")
print("\nNext → run 05_mapping_rules.py")
