"""
Visualisation functions for FL experiment results.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.model import create_model


CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]


# ---------- accuracy / loss comparison ----------------------------------------

def plot_accuracy_comparison(df_fedavg, df_qpso, save_path):
    """2×2 grid: global acc, global loss, FedAvg clients, QPSO clients."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("FedAvg vs QPSO-FL: Comprehensive Comparison",
                 fontsize=16, fontweight="bold")

    # (0,0) global accuracy
    ax = axes[0, 0]
    ax.plot(df_fedavg["round"], df_fedavg["global_test_acc"],
            label="FedAvg", lw=2, marker="o", ms=4, markevery=10)
    ax.plot(df_qpso["round"],  df_qpso["global_test_acc"],
            label="QPSO-FL", lw=2, marker="s", ms=4, markevery=10)
    ax.set(xlabel="Round", ylabel="Accuracy (%)",
           title="Global Test Accuracy", ylim=[0, 100])
    ax.legend(); ax.grid(alpha=0.3)

    # (0,1) global loss
    ax = axes[0, 1]
    ax.plot(df_fedavg["round"], df_fedavg["global_test_loss"], label="FedAvg", lw=2)
    ax.plot(df_qpso["round"],  df_qpso["global_test_loss"],  label="QPSO-FL", lw=2)
    ax.set(xlabel="Round", ylabel="Loss", title="Global Test Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # (1,0) FedAvg per-client
    ax = axes[1, 0]
    for i in range(1, 4):
        ax.plot(df_fedavg["round"], df_fedavg[f"client{i}_val_acc"],
                label=f"Client {i}", lw=2)
    ax.set(xlabel="Round", ylabel="Val Accuracy (%)",
           title="FedAvg: Per-Client Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    # (1,1) QPSO per-client
    ax = axes[1, 1]
    for i in range(1, 4):
        ax.plot(df_qpso["round"], df_qpso[f"client{i}_val_acc"],
                label=f"Client {i}", lw=2)
    ax.set(xlabel="Round", ylabel="Val Accuracy (%)",
           title="QPSO-FL: Per-Client Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {save_path}")


# ---------- confusion matrix --------------------------------------------------

def generate_confusion_matrix(model_path, test_loader, method_name,
                              device="cuda",
                              save_path=None):
    """Load best model, predict, plot heatmap, print classification report."""
    model = create_model(num_classes=3, device=device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _, pred = model(images).max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "Count"})
    plt.title(f"Confusion Matrix — {method_name}", fontsize=14, fontweight="bold")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n{method_name} — Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=4))
    return cm


# ---------- fairness bar chart ------------------------------------------------

def plot_client_fairness(df_fedavg, df_qpso, save_path):
    """Side-by-side bar chart of final per-client accuracy with std."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    labels = ["Client 1", "Client 2", "Client 3"]

    for ax, df, title in [
        (axes[0], df_fedavg, "FedAvg"),
        (axes[1], df_qpso,   "QPSO-FL"),
    ]:
        accs = [df[f"client{i}_val_acc"].iloc[-1] for i in range(1, 4)]
        ax.bar(labels, accs, color=colors, edgecolor="black", lw=1.5)
        ax.axhline(np.mean(accs), color="red", ls="--", lw=2, label="Mean")
        ax.set(ylabel="Final Val Accuracy (%)", ylim=[0, 100])
        ax.set_title(f"{title}  (σ = {np.std(accs):.2f}%)",
                     fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {save_path}")
