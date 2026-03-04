import json, os

def make_cell(cell_type, source):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell

def make_notebook(cells_data, filename):
    cells = [make_cell(t, s) for t, s in cells_data]
    # Remove trailing newline from last line of each cell
    for c in cells:
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created: {filename}")


# =====================================================================
# NOTEBOOK 1: DATA PREPARATION
# =====================================================================
nb1_cells = [
    ("markdown", """# Notebook 1: Data Preparation

**Settings:** GPU P100 · Persistence: Files only · Internet: ON

**Add Inputs:** Brain Tumor MRI Dataset (masoudnickparvar) + BRISC 2025 (briscdataset)"""),

    ("code", """import os, random, json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch

# Seed
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    device = torch.device("cpu"); print("CPU only")

# Directories
for d in ['data/processed/client1','data/processed/client2','data/processed/client3',
          'data/test_set','models','results/fedavg','results/qpso','results/plots','logs']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)
print("✅ Directories created")"""),

    ("code", """class BrainTumorPreprocessor:
    VALID_CLASSES = ["glioma", "meningioma", "pituitary"]
    CLASS_TO_IDX  = {"glioma": 0, "meningioma": 1, "pituitary": 2}

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def load_image(self, path):
        try:
            img = Image.open(path)
            if img.mode != "RGB": img = img.convert("RGB")
            img = img.resize(self.target_size, Image.LANCZOS)
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Error {path}: {e}"); return None

    def process_dataset(self, dataset_path, client_name,
                        save_dir="/kaggle/working/data/processed",
                        train_ratio=0.70, val_ratio=0.15):
        print(f"\\n{'='*60}\\nProcessing {client_name} <- {dataset_path}\\n{'='*60}")
        images, labels = [], []
        for cls_name in sorted(os.listdir(dataset_path)):
            cls_path = os.path.join(dataset_path, cls_name)
            if not os.path.isdir(cls_path): continue
            if cls_name.lower() not in self.VALID_CLASSES:
                print(f"  Skipping: {cls_name}"); continue
            label = self.CLASS_TO_IDX[cls_name.lower()]
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            print(f"  {cls_name}: {len(files)} images")
            for fname in tqdm(files, desc=f"    {cls_name}", leave=False):
                arr = self.load_image(os.path.join(cls_path, fname))
                if arr is not None: images.append(arr); labels.append(label)

        X, y = np.array(images), np.array(labels)
        print(f"\\n  Total: {len(X)}  Shape: {X.shape}  Classes: {np.bincount(y, minlength=3)}")

        test_ratio = 1.0 - train_ratio - val_ratio
        X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
        val_adj = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_adj, stratify=y_tmp, random_state=42)
        print(f"  Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

        out = os.path.join(save_dir, client_name)
        os.makedirs(out, exist_ok=True)
        for name, arr in [("X_train",X_train),("y_train",y_train),("X_val",X_val),("y_val",y_val),("X_test",X_test),("y_test",y_test)]:
            np.save(os.path.join(out, f"{name}.npy"), arr)
        print(f"  ✅ Saved -> {out}")

    @staticmethod
    def create_global_test_set(processed_dir="/kaggle/working/data/processed", save_dir="/kaggle/working/data/test_set"):
        Xs, ys = [], []
        for i in range(1, 4):
            d = os.path.join(processed_dir, f"client{i}")
            Xs.append(np.load(os.path.join(d, "X_test.npy")))
            ys.append(np.load(os.path.join(d, "y_test.npy")))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X_test.npy"), X)
        np.save(os.path.join(save_dir, "y_test.npy"), y)
        print(f"✅ Global test: {len(X)} images -> {save_dir}")

print("✅ Preprocessor defined")"""),

    ("markdown", """## Verify Dataset Paths
Update paths below if needed. Run `os.listdir("/kaggle/input")` to check."""),

    ("code", """# UPDATE THESE IF YOUR PATHS DIFFER!
CLIENT1_PATH = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing'
CLIENT2_PATH = '/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/train'
CLIENT3_PATH = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Training'

for name, path in [("Client1 (Masoud Test)", CLIENT1_PATH),
                   ("Client2 (BRISC)", CLIENT2_PATH),
                   ("Client3 (Masoud Train)", CLIENT3_PATH)]:
    print(f"\\n{name}: {path}")
    if os.path.exists(path):
        for d in sorted(os.listdir(path)):
            full = os.path.join(path, d)
            if os.path.isdir(full): print(f"  📂 {d}/ ({len(os.listdir(full))} files)")
    else:
        print("  ❌ PATH NOT FOUND!")"""),

    ("code", """prep = BrainTumorPreprocessor(target_size=(224, 224))
prep.process_dataset(CLIENT1_PATH, 'client1')
prep.process_dataset(CLIENT2_PATH, 'client2')
prep.process_dataset(CLIENT3_PATH, 'client3')"""),

    ("code", """BrainTumorPreprocessor.create_global_test_set()"""),

    ("code", """for c in ['client1', 'client2', 'client3']:
    d = f'/kaggle/working/data/processed/{c}'
    for f in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        arr = np.load(f'{d}/{f}.npy')
        print(f"  {c}/{f}: {arr.shape}")

gx = np.load('/kaggle/working/data/test_set/X_test.npy')
gy = np.load('/kaggle/working/data/test_set/y_test.npy')
print(f"\\nGlobal test: X={gx.shape} y={gy.shape} classes={np.bincount(gy)}")
print("\\n✅ Notebook 1 complete! Save Version -> Save & Run All")"""),
]

make_notebook(nb1_cells, "d:/Major_Project/FL_QPSO_FedAvg/Federated Learning QPSO/notebooks/notebook1_data_prep.ipynb")


# =====================================================================
# NOTEBOOK 2: TRAINING
# =====================================================================
nb2_cells = [
    ("markdown", """# Notebook 2: Training (FedAvg + QPSO)

**Settings:** GPU P100 · Persistence: Files only · Internet: ON

**Add Inputs:** Your Notebook 1 output (via + Add Input → Notebook Output tab)"""),

    ("code", """import os, sys, copy, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Seed
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu"); print("⚠️ CPU only")
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# Directories
for d in ['data/processed/client1','data/processed/client2','data/processed/client3',
          'data/test_set','models','results/fedavg','results/fedprox','results/qpso','results/plots','logs']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)

# ━━━━━━━ DATASET ━━━━━━━
class BrainTumorDataset(Dataset):
    _DEFAULT = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __init__(self, X, y, transform=None):
        self.X, self.y = X, y
        self.transform = transform or self._DEFAULT
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = Image.fromarray((self.X[idx]*255).astype(np.uint8))
        return self.transform(img), torch.tensor(self.y[idx], dtype=torch.long)

TRAIN_TF = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TEST_TF = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def create_data_loaders(processed_dir="/kaggle/working/data/processed",
                        test_dir="/kaggle/working/data/test_set", batch_size=32, num_workers=2):
    loaders = {}
    for i in range(1, 4):
        d = f"{processed_dir}/client{i}"
        ds_tr = BrainTumorDataset(np.load(f"{d}/X_train.npy"), np.load(f"{d}/y_train.npy"), TRAIN_TF)
        ds_va = BrainTumorDataset(np.load(f"{d}/X_val.npy"), np.load(f"{d}/y_val.npy"), TEST_TF)
        ds_te = BrainTumorDataset(np.load(f"{d}/X_test.npy"), np.load(f"{d}/y_test.npy"), TEST_TF)
        loaders[f"client{i}"] = {
            "train": DataLoader(ds_tr, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "val": DataLoader(ds_va, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            "test": DataLoader(ds_te, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            "train_size": len(ds_tr)}
        print(f"client{i}: Train={len(ds_tr)} Val={len(ds_va)} Test={len(ds_te)}")
    gds = BrainTumorDataset(np.load(f"{test_dir}/X_test.npy"), np.load(f"{test_dir}/y_test.npy"), TEST_TF)
    loaders["global_test"] = DataLoader(gds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Global test: {len(gds)}"); return loaders

# ━━━━━━━ MODEL ━━━━━━━
class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x): return self.model(x)

def create_model(num_classes=3, device="cuda"):
    m = BrainTumorResNet(num_classes).to(device)
    print(f"Model: ResNet-18 | Params: {sum(p.numel() for p in m.parameters()):,}")
    return m

# ━━━━━━━ CLIENT ━━━━━━━
class FederatedClient:
    def __init__(self, client_id, train_loader, val_loader, device="cuda"):
        self.client_id = client_id; self.train_loader = train_loader
        self.val_loader = val_loader; self.device = device
        self.model = None; self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.dataset_size = len(train_loader.dataset)
    def set_model(self, global_model): self.model = copy.deepcopy(global_model).to(self.device)
    def set_optimizer(self, lr=0.001): self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    def train_local(self, epochs=5, verbose=False):
        self.model.train(); ep_losses, ep_accs = [], []
        for ep in range(epochs):
            loss_sum, correct, total = 0.0, 0, 0
            for imgs, labs in self.train_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                self.optimizer.zero_grad(); out = self.model(imgs)
                loss = self.criterion(out, labs); loss.backward(); self.optimizer.step()
                loss_sum += loss.item(); _, pred = out.max(1)
                total += labs.size(0); correct += pred.eq(labs).sum().item()
            ep_losses.append(loss_sum/len(self.train_loader))
            ep_accs.append(100.*correct/total)
        return self.model.state_dict(), ep_losses, ep_accs
    def validate(self):
        self.model.eval(); loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in self.val_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                out = self.model(imgs); loss_sum += self.criterion(out, labs).item()
                _, pred = out.max(1); total += labs.size(0); correct += pred.eq(labs).sum().item()
        return loss_sum/len(self.val_loader), 100.*correct/total
    def get_dataset_size(self): return self.dataset_size

# ━━━━━━━ FEDPROX CLIENT ━━━━━━━
class FedProxClient(FederatedClient):
    def train_local(self, epochs=5, verbose=False, mu=0.01, global_params=None):
        self.model.train(); ep_losses, ep_accs = [], []
        for ep in range(epochs):
            loss_sum, correct, total = 0.0, 0, 0
            for imgs, labs in self.train_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                self.optimizer.zero_grad(); out = self.model(imgs)
                loss = self.criterion(out, labs)
                if global_params is not None:
                    prox = 0.0
                    for p_local, p_global in zip(self.model.parameters(), global_params):
                        prox += torch.norm(p_local - p_global.detach()) ** 2
                    loss = loss + (mu / 2.0) * prox
                loss.backward(); self.optimizer.step()
                loss_sum += loss.item(); _, pred = out.max(1)
                total += labs.size(0); correct += pred.eq(labs).sum().item()
            ep_losses.append(loss_sum/len(self.train_loader))
            ep_accs.append(100.*correct/total)
        return self.model.state_dict(), ep_losses, ep_accs

# ━━━━━━━ FEDAVG SERVER ━━━━━━━
class FedAvgServer:
    def __init__(self, global_model, clients, device="cuda"):
        self.global_model = global_model; self.clients = clients; self.device = device
        self.total_samples = sum(c.get_dataset_size() for c in clients)
        print(f"FedAvg Server | clients={len(clients)} total={self.total_samples}")
    def aggregate_weights(self, client_weights):
        first_sd = client_weights[0][0]; agg = {}
        for k in first_sd:
            if first_sd[k].is_floating_point():
                acc = torch.zeros_like(first_sd[k], dtype=torch.float32)
                for sd, n_k in client_weights:
                    acc = acc + sd[k].float() * (n_k / self.total_samples)
                agg[k] = acc.to(first_sd[k].dtype)
            else:
                agg[k] = first_sd[k].clone()
        return agg
    def evaluate_global_model(self, test_loader):
        self.global_model.eval(); crit = nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                out = self.global_model(imgs); loss_sum += crit(out, labs).item()
                _, pred = out.max(1); total += labs.size(0); correct += pred.eq(labs).sum().item()
        return 100.*correct/total, loss_sum/len(test_loader)
    def get_global_model(self): return copy.deepcopy(self.global_model)

# ━━━━━━━ QPSO SERVER ━━━━━━━
class QPSOServer:
    def __init__(self, global_model, clients, device="cuda", beta=0.7):
        self.global_model = global_model; self.clients = clients
        self.device = device; self.beta = beta
        self.personal_best = {}; self.personal_best_scores = {}
        self.global_best = None; self.global_best_score = 0.0; self.mean_best = None
        print(f"QPSO Server | clients={len(clients)} β={beta}")
    def initialize_particles(self):
        state = copy.deepcopy(self.global_model.state_dict())
        for c in self.clients:
            self.personal_best[c.client_id] = copy.deepcopy(state)
            self.personal_best_scores[c.client_id] = 0.0
        self.global_best = copy.deepcopy(state); self.global_best_score = 0.0
        print("✅ QPSO particles initialised")
    def update_personal_best(self, cid, weights, val_acc):
        if val_acc > self.personal_best_scores[cid]:
            self.personal_best[cid] = copy.deepcopy(weights)
            self.personal_best_scores[cid] = val_acc; return True
        return False
    def update_global_best(self, cid, val_acc):
        if val_acc > self.global_best_score:
            self.global_best = copy.deepcopy(self.personal_best[cid])
            self.global_best_score = val_acc; return True
        return False
    def calculate_mean_best(self):
        first_id = self.clients[0].client_id
        self.mean_best = {k: torch.zeros_like(v, dtype=torch.float32)
                          for k, v in self.personal_best[first_id].items()}
        for c in self.clients:
            for k in self.mean_best: self.mean_best[k] += self.personal_best[c.client_id][k].float()
        for k in self.mean_best: self.mean_best[k] /= len(self.clients)
    def qpso_aggregate(self, client_weights_list):
        for cid, w, acc in client_weights_list:
            pb = self.update_personal_best(cid, w, acc)
            gb = self.update_global_best(cid, acc)
            if pb: print(f"  {cid}: pbest ↑ {acc:.2f}%")
            if gb: print(f"  {cid}: gbest ↑ {acc:.2f}%")
        self.calculate_mean_best(); agg = copy.deepcopy(self.global_best)
        for k in agg:
            if not agg[k].is_floating_point(): continue
            pbest_sum = torch.zeros_like(agg[k], dtype=torch.float32)
            for c in self.clients: pbest_sum += self.personal_best[c.client_id][k].float()
            phi = torch.rand_like(agg[k].float())
            u = torch.rand_like(agg[k].float()).clamp(min=0.3, max=1.0)
            p = phi*(pbest_sum/len(self.clients)) + (1-phi)*self.global_best[k].float()
            sign = torch.where(torch.rand_like(agg[k].float())<0.5,
                               torch.ones_like(agg[k].float()), -torch.ones_like(agg[k].float()))
            perturbation = self.beta * torch.abs(self.mean_best[k].float() - agg[k].float()) * torch.log(1.0/u)
            perturbation = perturbation.clamp(-0.1, 0.1)
            new_val = p + sign * perturbation
            agg[k] = new_val.to(self.global_best[k].dtype)
        return agg
    def evaluate_global_model(self, test_loader):
        self.global_model.eval(); crit = nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                out = self.global_model(imgs); loss_sum += crit(out, labs).item()
                _, pred = out.max(1); total += labs.size(0); correct += pred.eq(labs).sum().item()
        return 100.*correct/total, loss_sum/len(test_loader)
    def get_global_model(self): return copy.deepcopy(self.global_model)

# ━━━━━━━ TRAINERS ━━━━━━━
def train_fedavg(server, clients, global_test_loader, num_rounds=100, local_epochs=5,
                 learning_rate=0.001, save_every=10, save_dir="/kaggle/working", verbose=False):
    print("="*80+"\\n  FEDAVG TRAINING START\\n"+"="*80)
    history = {"round":[],"global_test_acc":[],"global_test_loss":[],
               "client1_val_acc":[],"client2_val_acc":[],"client3_val_acc":[],"round_time":[]}
    best_acc = 0.0
    for rnd in range(1, num_rounds+1):
        t0 = time.time(); print(f"\\n{'='*60} ROUND {rnd}/{num_rounds}"); cw = []
        for client in clients:
            print(f"  {client.client_id} training...", end=" ")
            client.set_model(server.get_global_model()); client.set_optimizer(learning_rate)
            w,_,_ = client.train_local(local_epochs, verbose=verbose)
            vl, va = client.validate(); print(f"val={va:.2f}%")
            history[f"{client.client_id}_val_acc"].append(va)
            cw.append((w, client.get_dataset_size()))
        agg = server.aggregate_weights(cw); server.global_model.load_state_dict(agg)
        g_acc, g_loss = server.evaluate_global_model(global_test_loader); dt = time.time()-t0
        history["round"].append(rnd); history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss); history["round_time"].append(dt)
        print(f"  Global: acc={g_acc:.2f}% loss={g_loss:.4f} ({dt:.1f}s)")
        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(), f"{save_dir}/models/fedavg_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(), f"{save_dir}/models/fedavg_round_{rnd}.pth")
        pd.DataFrame(history).to_csv(f"{save_dir}/results/fedavg/metrics.csv", index=False)
    print(f"\\n{'='*80}\\n  FEDAVG DONE — best {best_acc:.2f}%\\n{'='*80}"); return history

def train_fedprox(server, clients, global_test_loader, num_rounds=100, local_epochs=5,
                  learning_rate=0.001, mu=0.01, save_every=10, save_dir="/kaggle/working", verbose=False):
    print("="*80+"\\n  FEDPROX TRAINING START (μ="+str(mu)+")\\n"+"="*80)
    history = {"round":[],"global_test_acc":[],"global_test_loss":[],
               "client1_val_acc":[],"client2_val_acc":[],"client3_val_acc":[],"round_time":[]}
    best_acc = 0.0
    for rnd in range(1, num_rounds+1):
        t0 = time.time(); print(f"\\n{'='*60} ROUND {rnd}/{num_rounds}"); cw = []
        global_params = list(server.global_model.parameters())
        for client in clients:
            print(f"  {client.client_id} training...", end=" ")
            client.set_model(server.get_global_model()); client.set_optimizer(learning_rate)
            w,_,_ = client.train_local(local_epochs, verbose=verbose, mu=mu, global_params=global_params)
            vl, va = client.validate(); print(f"val={va:.2f}%")
            history[f"{client.client_id}_val_acc"].append(va)
            cw.append((w, client.get_dataset_size()))
        agg = server.aggregate_weights(cw); server.global_model.load_state_dict(agg)
        g_acc, g_loss = server.evaluate_global_model(global_test_loader); dt = time.time()-t0
        history["round"].append(rnd); history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss); history["round_time"].append(dt)
        print(f"  Global: acc={g_acc:.2f}% loss={g_loss:.4f} ({dt:.1f}s)")
        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(), f"{save_dir}/models/fedprox_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(), f"{save_dir}/models/fedprox_round_{rnd}.pth")
        pd.DataFrame(history).to_csv(f"{save_dir}/results/fedprox/metrics.csv", index=False)
    print(f"\\n{'='*80}\\n  FEDPROX DONE — best {best_acc:.2f}%\\n{'='*80}"); return history

def train_qpso(server, clients, global_test_loader, num_rounds=100, local_epochs=5,
               learning_rate=0.001, save_every=10, save_dir="/kaggle/working", verbose=False):
    print("="*80+"\\n  QPSO-FL TRAINING START\\n"+"="*80); server.initialize_particles()
    history = {"round":[],"global_test_acc":[],"global_test_loss":[],"global_best_score":[],
               "client1_val_acc":[],"client2_val_acc":[],"client3_val_acc":[],
               "client1_pbest_score":[],"client2_pbest_score":[],"client3_pbest_score":[],"round_time":[]}
    best_acc = 0.0
    for rnd in range(1, num_rounds+1):
        t0 = time.time(); print(f"\\n{'='*60} ROUND {rnd}/{num_rounds}"); cw = []
        for client in clients:
            print(f"  {client.client_id} training...", end=" ")
            client.set_model(server.get_global_model()); client.set_optimizer(learning_rate)
            w,_,_ = client.train_local(local_epochs, verbose=verbose)
            vl, va = client.validate(); print(f"val={va:.2f}%")
            history[f"{client.client_id}_val_acc"].append(va)
            cw.append((client.client_id, w, va))
        print("  QPSO aggregating..."); agg = server.qpso_aggregate(cw)
        server.global_model.load_state_dict(agg)
        for c in clients: history[f"{c.client_id}_pbest_score"].append(server.personal_best_scores[c.client_id])
        history["global_best_score"].append(server.global_best_score)
        g_acc, g_loss = server.evaluate_global_model(global_test_loader); dt = time.time()-t0
        history["round"].append(rnd); history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss); history["round_time"].append(dt)
        print(f"  Global: acc={g_acc:.2f}% loss={g_loss:.4f} gbest={server.global_best_score:.2f}% ({dt:.1f}s)")
        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(), f"{save_dir}/models/qpso_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(), f"{save_dir}/models/qpso_round_{rnd}.pth")
        pd.DataFrame(history).to_csv(f"{save_dir}/results/qpso/metrics.csv", index=False)
    print(f"\\n{'='*80}\\n  QPSO-FL DONE — best {best_acc:.2f}% gbest {server.global_best_score:.2f}%\\n{'='*80}")
    return history

print("✅ All code loaded!")"""),

    ("markdown", """## Copy Preprocessed Data from Notebook 1"""),

    ("code", """import shutil

# Exact path from your Notebook 1 output dataset
DATA_SRC = '/kaggle/input/datasets/divyanshtejaedla/fl-dataset-brain-tumour-classification/data'

if os.path.exists(DATA_SRC):
    shutil.copytree(DATA_SRC, '/kaggle/working/data', dirs_exist_ok=True)
    print(f"✅ Data copied from {DATA_SRC}")
else:
    # Fallback: search all inputs
    found = False
    for search in os.listdir('/kaggle/input'):
        data_src = f'/kaggle/input/{search}/data'
        if os.path.exists(data_src):
            shutil.copytree(data_src, '/kaggle/working/data', dirs_exist_ok=True)
            print(f"✅ Data copied from {data_src}")
            found = True; break
    if not found:
        print("❌ Data not found! Your inputs:")
        for d in os.listdir('/kaggle/input'): print(f"  {d}")"""),

    ("code", """data_loaders = create_data_loaders(batch_size=32, num_workers=2)"""),

    ("code", """# Quick model test
model = create_model(num_classes=3, device=device)
x = torch.randn(2, 3, 224, 224).to(device)
print(f"Output: {model(x).shape}")  # (2, 3)
del model, x; torch.cuda.empty_cache()"""),

    ("markdown", """## FedAvg Training
⚠️ Set `num_rounds=5` for a quick test first!"""),

    ("code", """fedavg_model = create_model(num_classes=3, device=device)
fedavg_clients = [FederatedClient(f'client{i}', data_loaders[f'client{i}']['train'],
                  data_loaders[f'client{i}']['val'], device=device) for i in range(1, 4)]
fedavg_server = FedAvgServer(fedavg_model, fedavg_clients, device=device)

fedavg_history = train_fedavg(
    server=fedavg_server, clients=fedavg_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=100, local_epochs=5, learning_rate=0.001,
    save_every=10, verbose=False)"""),

    ("markdown", """## FedProx Training
⚠️ Set `num_rounds=5` for a quick test first!"""),

    ("code", """torch.cuda.empty_cache()

fedprox_model = create_model(num_classes=3, device=device)
fedprox_clients = [FedProxClient(f'client{i}', data_loaders[f'client{i}']['train'],
                   data_loaders[f'client{i}']['val'], device=device) for i in range(1, 4)]
fedprox_server = FedAvgServer(fedprox_model, fedprox_clients, device=device)

fedprox_history = train_fedprox(
    server=fedprox_server, clients=fedprox_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=100, local_epochs=5, learning_rate=0.001,
    mu=0.01, save_every=10, verbose=False)"""),

    ("markdown", """## QPSO Training
⚠️ Set `num_rounds=5` for a quick test first!"""),

    ("code", """torch.cuda.empty_cache()

qpso_model = create_model(num_classes=3, device=device)
qpso_clients = [FederatedClient(f'client{i}', data_loaders[f'client{i}']['train'],
                data_loaders[f'client{i}']['val'], device=device) for i in range(1, 4)]
qpso_server = QPSOServer(qpso_model, qpso_clients, device=device, beta=0.7)

qpso_history = train_qpso(
    server=qpso_server, clients=qpso_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=100, local_epochs=5, learning_rate=0.001,
    save_every=10, verbose=False)

print("\\n✅ Notebook 2 complete! Save Version -> Save & Run All")""")
]

make_notebook(nb2_cells, "d:/Major_Project/FL_QPSO_FedAvg/Federated Learning QPSO/notebooks/notebook2_training.ipynb")


# =====================================================================
# NOTEBOOK 3: EVALUATION
# =====================================================================
nb3_cells = [
    ("markdown", """# Notebook 3: Evaluation & Plots

**Settings:** GPU P100

**Add Inputs:** Notebook 1 output + Notebook 2 output (via Notebook Output tab)"""),

    ("code", """import os, copy, json, random
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats

random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
for d in ['models','results/fedavg','results/fedprox','results/qpso','results/plots']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)

class BrainTumorDataset(Dataset):
    _TF = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = Image.fromarray((self.X[idx]*255).astype(np.uint8))
        return self._TF(img), torch.tensor(self.y[idx], dtype=torch.long)

def create_data_loaders(pdir="/kaggle/working/data/processed", tdir="/kaggle/working/data/test_set", bs=32):
    loaders = {}
    for i in range(1, 4):
        d = f"{pdir}/client{i}"
        for s in ['train','val','test']:
            ds = BrainTumorDataset(np.load(f"{d}/X_{s}.npy"), np.load(f"{d}/y_{s}.npy"))
            loaders.setdefault(f"client{i}", {})[s] = DataLoader(ds, bs, shuffle=(s=='train'), num_workers=2, pin_memory=True)
        loaders[f"client{i}"]["train_size"] = len(np.load(f"{d}/X_train.npy"))
    gds = BrainTumorDataset(np.load(f"{tdir}/X_test.npy"), np.load(f"{tdir}/y_test.npy"))
    loaders["global_test"] = DataLoader(gds, bs, shuffle=False, num_workers=2, pin_memory=True)
    return loaders

class BrainTumorResNet(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, n)
    def forward(self, x): return self.model(x)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]

def plot_accuracy_comparison(dfs, names, save_path):
    n = len(dfs)
    fig, axes = plt.subplots(2, max(n,2), figsize=(8*max(n,2), 12))
    fig.suptitle("FL Aggregation Comparison", fontsize=16, fontweight="bold")
    colors = ['#1f77b4','#ff7f0e','#2ca02c']
    ax = axes[0,0]
    for df, name, c in zip(dfs, names, colors):
        ax.plot(df["round"], df["global_test_acc"], label=name, lw=2, color=c)
    ax.set(xlabel="Round", ylabel="Acc (%)", title="Global Test Accuracy", ylim=[0,100]); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0,1]
    for df, name, c in zip(dfs, names, colors):
        ax.plot(df["round"], df["global_test_loss"], label=name, lw=2, color=c)
    ax.set(xlabel="Round", ylabel="Loss", title="Global Test Loss"); ax.legend(); ax.grid(alpha=0.3)
    for idx, (df, name) in enumerate(zip(dfs, names)):
        ax = axes[1, idx]
        for i in range(1,4): ax.plot(df["round"], df[f"client{i}_val_acc"], label=f"Client {i}", lw=2)
        ax.set(xlabel="Round", ylabel="Val Acc (%)", title=f"{name}: Per-Client"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.show()
    print(f"✅ Saved -> {save_path}")

def generate_confusion_matrix(model_path, test_loader, method_name, device="cuda", save_path=None):
    model = BrainTumorResNet(3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            _, pred = model(imgs.to(device)).max(1)
            preds.extend(pred.cpu().numpy()); labels.extend(labs.numpy())
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {method_name}", fontsize=14, fontweight="bold")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\\n{method_name} Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

def plot_roc_auc(model_paths, method_names, test_loader, device="cuda", save_path=None):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    n = len(model_paths)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    colors_cls = ['#e74c3c', '#3498db', '#2ecc71']
    for ax, mp, mname in zip(axes, model_paths, method_names):
        model = BrainTumorResNet(3).to(device)
        model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
        model.eval(); all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labs in test_loader:
                out = model(imgs.to(device))
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs.cpu().numpy()); all_labels.extend(labs.numpy())
        y_prob = np.vstack(all_probs); y_true = np.array(all_labels)
        y_bin = label_binarize(y_true, classes=[0,1,2])
        for i, (cls, c) in enumerate(zip(CLASS_NAMES, colors_cls)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{cls} (AUC={roc_auc:.4f})")
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        ax.plot(fpr_micro, tpr_micro, 'k--', lw=2, label=f"Micro-avg (AUC={auc(fpr_micro,tpr_micro):.4f})")
        ax.plot([0,1],[0,1],'gray',ls=':', lw=1)
        ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {mname}")
        ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(); print(f"✅ ROC-AUC saved -> {save_path}")

def plot_client_fairness(dfs, names, save_path):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5)); colors = ["#FF6B6B","#4ECDC4","#45B7D1"]
    if n == 1: axes = [axes]
    for ax, df, t in zip(axes, dfs, names):
        accs = [df[f"client{i}_val_acc"].iloc[-1] for i in range(1,4)]
        ax.bar(["C1","C2","C3"], accs, color=colors, edgecolor="black")
        ax.axhline(np.mean(accs), color="red", ls="--", lw=2, label="Mean")
        ax.set(ylabel="Final Val Acc (%)", ylim=[0,100])
        ax.set_title(f"{t} (σ={np.std(accs):.2f}%)", fontsize=13, fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.show()

def convergence_metrics(df, target=80.0):
    reached = df[df["global_test_acc"]>=target]
    return {"final_acc": round(df["global_test_acc"].iloc[-1],2),
            "best_acc": round(df["global_test_acc"].max(),2),
            "round_to_target": int(reached["round"].min()) if len(reached) else None,
            "avg_round_time_s": round(df["round_time"].mean(),2),
            "total_time_min": round(df["round_time"].sum()/60,2)}

def statistical_analysis(df_fa, df_qp):
    a, b = df_fa["global_test_acc"].values, df_qp["global_test_acc"].values
    n = min(len(a),len(b)); a, b = a[:n], b[:n]
    t_stat, p_val = stats.ttest_rel(b, a); d = (b-a).mean()/((b-a).std(ddof=1)+1e-8)
    return {"t_stat": round(float(t_stat),4), "p_value": float(p_val),
            "cohens_d": round(float(d),4), "significant": bool(p_val<0.05)}

def build_comparison_df(*dfs_and_names):
    rows = []
    cols = ["Metric"] + [n for _, n in dfs_and_names]
    for _, name in dfs_and_names:
        pass  # just to get names
    metrics_list = []
    for df, name in dfs_and_names:
        m = convergence_metrics(df)
        m["client_std"] = round(float(np.std([df[f"client{i}_val_acc"].iloc[-1] for i in range(1,4)])),2)
        metrics_list.append(m)
    metric_keys = [("Final Acc (%)","final_acc"),("Best Acc (%)","best_acc"),
                   ("Rounds to 80%","round_to_target"),("Avg Round (s)","avg_round_time_s"),
                   ("Total Time (min)","total_time_min"),("Client Std Dev","client_std")]
    for label, key in metric_keys:
        row = [label] + [m[key] if m[key] is not None else "N/A" for m in metrics_list]
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

print("✅ All code loaded")"""),

    ("code", """import shutil

DATA_SRC_NB1 = '/kaggle/input/datasets/divyanshtejaedla/fl-dataset-brain-tumour-classification/data'
DATA_SRC_NB2 = '/kaggle/input/datasets/divyanshtejaedla/fl-qpso-training-results'
print("Starting to load data...")

# 1. Load Data from Notebook 1
if os.path.exists(DATA_SRC_NB1):
    shutil.copytree(DATA_SRC_NB1, '/kaggle/working/data', dirs_exist_ok=True)
    print(f"✅ Data copied from {DATA_SRC_NB1}")
else:
    print(f"⚠️ Could not find precise data path: {DATA_SRC_NB1}")

# 2. Load Models/Results from Notebook 2
if os.path.exists(DATA_SRC_NB2):
    for sub in ['models', 'results', 'logs']:
        sub_src = os.path.join(DATA_SRC_NB2, sub)
        if os.path.exists(sub_src):
            shutil.copytree(sub_src, f'/kaggle/working/{sub}', dirs_exist_ok=True)
            print(f"✅ Copied {sub} from {DATA_SRC_NB2}")
else:
    print(f"⚠️ Could not find precise models path: {DATA_SRC_NB2}")

# 3. Fallback: Search all inputs just in case
print("\\nScanning /kaggle/input for any missing directories...")
for base_name in os.listdir('/kaggle/input'):
    if 'divyanshtejaedla' in base_name: continue # Already checked
    src = f'/kaggle/input/{base_name}'
    for sub in ['data','models','results','logs']:
        sub_src = os.path.join(src, sub)
        if os.path.exists(sub_src):
            shutil.copytree(sub_src, f'/kaggle/working/{sub}', dirs_exist_ok=True)
            print(f"  Copied fallback {base_name}/{sub}")

print("\\nInitializing Data Loaders...")
data_loaders = create_data_loaders()
print("✅ Data loaded")"""),

    ("code", """df_fa = pd.read_csv('/kaggle/working/results/fedavg/metrics.csv')
df_fp = pd.read_csv('/kaggle/working/results/fedprox/metrics.csv')
df_qp = pd.read_csv('/kaggle/working/results/qpso/metrics.csv')
comp = build_comparison_df((df_fa,"FedAvg"),(df_fp,"FedProx"),(df_qp,"QPSO-FL"))
print(comp.to_string(index=False))
comp.to_csv('/kaggle/working/results/final_comparison.csv', index=False)"""),

    ("code", """plot_accuracy_comparison([df_fa, df_fp, df_qp], ['FedAvg','FedProx','QPSO-FL'], '/kaggle/working/results/plots/comprehensive_comparison.png')
generate_confusion_matrix('/kaggle/working/models/fedavg_best.pth', data_loaders['global_test'], 'FedAvg', device=device, save_path='/kaggle/working/results/plots/cm_fedavg.png')
generate_confusion_matrix('/kaggle/working/models/fedprox_best.pth', data_loaders['global_test'], 'FedProx', device=device, save_path='/kaggle/working/results/plots/cm_fedprox.png')
generate_confusion_matrix('/kaggle/working/models/qpso_best.pth', data_loaders['global_test'], 'QPSO-FL', device=device, save_path='/kaggle/working/results/plots/cm_qpso.png')
plot_client_fairness([df_fa, df_fp, df_qp], ['FedAvg','FedProx','QPSO-FL'], '/kaggle/working/results/plots/client_fairness.png')
plot_roc_auc(
    ['/kaggle/working/models/fedavg_best.pth', '/kaggle/working/models/fedprox_best.pth', '/kaggle/working/models/qpso_best.pth'],
    ['FedAvg', 'FedProx', 'QPSO-FL'], data_loaders['global_test'], device=device,
    save_path='/kaggle/working/results/plots/roc_auc.png')"""),

    ("code", """st_fa_qp = statistical_analysis(df_fa, df_qp)
st_fa_fp = statistical_analysis(df_fa, df_fp)
print("FedAvg vs QPSO:", json.dumps(st_fa_qp, indent=2))
print("FedAvg vs FedProx:", json.dumps(st_fa_fp, indent=2))
summary = {"fedavg": convergence_metrics(df_fa), "fedprox": convergence_metrics(df_fp),
           "qpso": convergence_metrics(df_qp),
           "stats_fedavg_vs_qpso": st_fa_qp, "stats_fedavg_vs_fedprox": st_fa_fp}
with open('/kaggle/working/results/executive_summary.json','w') as f: json.dump(summary, f, indent=2)
print("✅ Summary saved")
print("\\nLaTeX Table:")
print(comp.to_latex(index=False, float_format="%.2f", caption="FedAvg vs FedProx vs QPSO-FL", label="tab:comparison"))
print("\\n✅ Notebook 3 complete!")"""),
]

make_notebook(nb3_cells, "d:/Major_Project/FL_QPSO_FedAvg/Federated Learning QPSO/notebooks/notebook3_evaluation.ipynb")
