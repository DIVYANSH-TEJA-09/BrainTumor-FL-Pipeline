########################################################################
# NOTEBOOK 3: EVALUATION & PLOTS
#
# Kaggle Settings:
#   Accelerator: GPU P100
#   Persistence: Files only
#
# Input Datasets (add via + Add Input):
#   - Notebook 1 output (preprocessed data)
#   - Notebook 2 output (training results)
#   OR use "Notebook Output" tab to add them
#
# Paste each "# ─── Cell N ───" section into a separate Kaggle cell.
########################################################################


# ─── Cell 1: Setup + All Code ────────────────────────────────────────

import os, sys, copy, json, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats

# ── Seed + GPU ──
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")

# ── Dirs ──
for d in ['models', 'results/fedavg', 'results/qpso', 'results/plots']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)

# ── Dataset + DataLoader ──
class BrainTumorDataset(Dataset):
    _TF = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = Image.fromarray((self.X[idx]*255).astype(np.uint8))
        return self._TF(img), torch.tensor(self.y[idx], dtype=torch.long)

def create_data_loaders(processed_dir="/kaggle/working/data/processed",
                        test_dir="/kaggle/working/data/test_set", bs=32):
    loaders = {}
    for i in range(1, 4):
        d = f"{processed_dir}/client{i}"
        for split in ['train', 'val', 'test']:
            X = np.load(f"{d}/X_{split}.npy")
            y = np.load(f"{d}/y_{split}.npy")
            ds = BrainTumorDataset(X, y)
            loaders.setdefault(f"client{i}", {})[split] = \
                DataLoader(ds, bs, shuffle=(split=='train'), num_workers=2, pin_memory=True)
        loaders[f"client{i}"]["train_size"] = len(np.load(f"{d}/X_train.npy"))
    gds = BrainTumorDataset(np.load(f"{test_dir}/X_test.npy"),
                            np.load(f"{test_dir}/y_test.npy"))
    loaders["global_test"] = DataLoader(gds, bs, shuffle=False, num_workers=2, pin_memory=True)
    return loaders

# ── Model ──
class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_accuracy_comparison(df_fa, df_qp, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("FedAvg vs QPSO-FL", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df_fa["round"], df_fa["global_test_acc"], label="FedAvg", lw=2, marker="o", ms=3, markevery=10)
    ax.plot(df_qp["round"], df_qp["global_test_acc"], label="QPSO-FL", lw=2, marker="s", ms=3, markevery=10)
    ax.set(xlabel="Round", ylabel="Accuracy (%)", title="Global Test Accuracy", ylim=[0, 100])
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df_fa["round"], df_fa["global_test_loss"], label="FedAvg", lw=2)
    ax.plot(df_qp["round"], df_qp["global_test_loss"], label="QPSO-FL", lw=2)
    ax.set(xlabel="Round", ylabel="Loss", title="Global Test Loss")
    ax.legend(); ax.grid(alpha=0.3)

    for idx, (df, title) in enumerate([(df_fa, "FedAvg"), (df_qp, "QPSO-FL")]):
        ax = axes[1, idx]
        for i in range(1, 4):
            ax.plot(df["round"], df[f"client{i}_val_acc"], label=f"Client {i}", lw=2)
        ax.set(xlabel="Round", ylabel="Val Acc (%)", title=f"{title}: Per-Client")
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {save_path}")


def generate_confusion_matrix(model_path, test_loader, method_name,
                              device="cuda", save_path=None):
    model = BrainTumorResNet(3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            _, pred = model(imgs.to(device)).max(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(labs.numpy())

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {method_name}", fontsize=14, fontweight="bold")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n{method_name} Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    return cm


def plot_client_fairness(df_fa, df_qp, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for ax, df, title in [(axes[0], df_fa, "FedAvg"), (axes[1], df_qp, "QPSO-FL")]:
        accs = [df[f"client{i}_val_acc"].iloc[-1] for i in range(1, 4)]
        ax.bar(["Client 1","Client 2","Client 3"], accs, color=colors, edgecolor="black")
        ax.axhline(np.mean(accs), color="red", ls="--", lw=2, label="Mean")
        ax.set(ylabel="Final Val Acc (%)", ylim=[0, 100])
        ax.set_title(f"{title} (σ={np.std(accs):.2f}%)", fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def convergence_metrics(df, target=80.0):
    final = df["global_test_acc"].iloc[-1]
    best  = df["global_test_acc"].max()
    reached = df[df["global_test_acc"] >= target]
    r2t = int(reached["round"].min()) if len(reached) else None
    return {"final_acc": round(final,2), "best_acc": round(best,2),
            "round_to_target": r2t,
            "avg_round_time_s": round(df["round_time"].mean(),2),
            "total_time_min": round(df["round_time"].sum()/60,2)}


def statistical_analysis(df_fa, df_qp):
    a, b = df_fa["global_test_acc"].values, df_qp["global_test_acc"].values
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    t_stat, p_val = stats.ttest_rel(b, a)
    d = (b - a).mean() / ((b - a).std(ddof=1) + 1e-8)
    return {"t_stat": round(float(t_stat),4), "p_value": float(p_val),
            "cohens_d": round(float(d),4), "significant": bool(p_val < 0.05),
            "mean_improvement": round(float((b-a).mean()),4)}


def build_comparison_df(df_fa, df_qp):
    fa, qp = convergence_metrics(df_fa), convergence_metrics(df_qp)
    fa_std = np.std([df_fa[f"client{i}_val_acc"].iloc[-1] for i in range(1,4)])
    qp_std = np.std([df_qp[f"client{i}_val_acc"].iloc[-1] for i in range(1,4)])
    rows = [
        ("Final Global Acc (%)", fa["final_acc"], qp["final_acc"]),
        ("Best Global Acc (%)",  fa["best_acc"],  qp["best_acc"]),
        ("Rounds to 80%", fa["round_to_target"] or "N/A", qp["round_to_target"] or "N/A"),
        ("Avg Round Time (s)", fa["avg_round_time_s"], qp["avg_round_time_s"]),
        ("Total Time (min)", fa["total_time_min"], qp["total_time_min"]),
        ("Client Acc Std Dev", round(float(fa_std),2), round(float(qp_std),2)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "FedAvg", "QPSO-FL"])


def generate_latex_table(df, caption="FedAvg vs QPSO-FL"):
    return df.to_latex(index=False, float_format="%.2f",
                       caption=caption, label="tab:comparison")

print("✅ All code loaded")


# ─── Cell 2: Copy Data & Results ─────────────────────────────────────

import shutil

for base_name in os.listdir('/kaggle/input'):
    src = f'/kaggle/input/{base_name}'
    # Copy only data / models / results — NOT src
    for sub in ['data', 'models', 'results', 'logs']:
        sub_src = os.path.join(src, sub)
        if os.path.exists(sub_src):
            shutil.copytree(sub_src, f'/kaggle/working/{sub}', dirs_exist_ok=True)
            print(f"  Copied {base_name}/{sub}")

data_loaders = create_data_loaders()
print("✅ Data loaded")


# ─── Cell 3: Comparison Table ─────────────────────────────────────────

df_fa = pd.read_csv('/kaggle/working/results/fedavg/metrics.csv')
df_qp = pd.read_csv('/kaggle/working/results/qpso/metrics.csv')

comp = build_comparison_df(df_fa, df_qp)
print(comp.to_string(index=False))
comp.to_csv('/kaggle/working/results/final_comparison.csv', index=False)


# ─── Cell 4: All Plots ───────────────────────────────────────────────

plot_accuracy_comparison(df_fa, df_qp,
    '/kaggle/working/results/plots/comprehensive_comparison.png')

generate_confusion_matrix('/kaggle/working/models/fedavg_best.pth',
    data_loaders['global_test'], 'FedAvg', device=device,
    save_path='/kaggle/working/results/plots/cm_fedavg.png')

generate_confusion_matrix('/kaggle/working/models/qpso_best.pth',
    data_loaders['global_test'], 'QPSO-FL', device=device,
    save_path='/kaggle/working/results/plots/cm_qpso.png')

plot_client_fairness(df_fa, df_qp,
    '/kaggle/working/results/plots/client_fairness.png')


# ─── Cell 5: Statistical Analysis ────────────────────────────────────

st = statistical_analysis(df_fa, df_qp)
print("Statistical Test:", json.dumps(st, indent=2))

summary = {
    "fedavg": convergence_metrics(df_fa),
    "qpso": convergence_metrics(df_qp),
    "improvement": round(convergence_metrics(df_qp)["best_acc"]
                         - convergence_metrics(df_fa)["best_acc"], 2),
    "statistical_test": st,
}
with open('/kaggle/working/results/executive_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✅ Summary saved")

print("\nLaTeX Table:")
print(generate_latex_table(comp))

print("\n✅ Notebook 3 complete! All results saved in /kaggle/working/results/")
