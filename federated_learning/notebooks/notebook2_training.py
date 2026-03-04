########################################################################
# NOTEBOOK 2: TRAINING (FedAvg + QPSO)
#
# Kaggle Settings:
#   Accelerator: GPU P100
#   Persistence: Files only
#   Internet: ON
#
# Input Datasets (add via + Add Input):
#   - Your Notebook 1 output (fl-qpso-preprocessed-data)
#     OR add via "Notebook Output" tab → search "Data Preparation"
#
# Paste each "# ─── Cell N ───" section into a separate Kaggle cell.
########################################################################


# ─── Cell 1: Setup + All Source Code ─────────────────────────────────

import os, sys, copy, time, random, json
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

# ── Seed ──
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ── GPU ──
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ CPU only")

# ── Directories ──
for d in ['data/processed/client1', 'data/processed/client2',
          'data/processed/client3', 'data/test_set',
          'models', 'results/fedavg', 'results/qpso', 'results/plots', 'logs']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BrainTumorDataset(Dataset):
    _DEFAULT = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    def __init__(self, X, y, transform=None):
        self.X, self.y = X, y
        self.transform = transform or self._DEFAULT
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = Image.fromarray((self.X[idx]*255).astype(np.uint8))
        return self.transform(img), torch.tensor(self.y[idx], dtype=torch.long)


TRAIN_TF = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def create_data_loaders(processed_dir="/kaggle/working/data/processed",
                        test_dir="/kaggle/working/data/test_set",
                        batch_size=32, num_workers=2):
    loaders = {}
    for i in range(1, 4):
        d = f"{processed_dir}/client{i}"
        ds_tr = BrainTumorDataset(np.load(f"{d}/X_train.npy"),
                                  np.load(f"{d}/y_train.npy"), TRAIN_TF)
        ds_va = BrainTumorDataset(np.load(f"{d}/X_val.npy"),
                                  np.load(f"{d}/y_val.npy"), TEST_TF)
        ds_te = BrainTumorDataset(np.load(f"{d}/X_test.npy"),
                                  np.load(f"{d}/y_test.npy"), TEST_TF)
        loaders[f"client{i}"] = {
            "train": DataLoader(ds_tr, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
            "val":   DataLoader(ds_va, batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
            "test":  DataLoader(ds_te, batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
            "train_size": len(ds_tr),
        }
        print(f"client{i}: Train={len(ds_tr)} Val={len(ds_va)} Test={len(ds_te)}")

    gds = BrainTumorDataset(np.load(f"{test_dir}/X_test.npy"),
                            np.load(f"{test_dir}/y_test.npy"), TEST_TF)
    loaders["global_test"] = DataLoader(gds, batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=True)
    print(f"Global test: {len(gds)}")
    return loaders


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

def create_model(num_classes=3, device="cuda"):
    m = BrainTumorResNet(num_classes).to(device)
    total = sum(p.numel() for p in m.parameters())
    print(f"Model: ResNet-18 | Params: {total:,} | ~{total*4/1e6:.1f} MB")
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLIENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FederatedClient:
    def __init__(self, client_id, train_loader, val_loader, device="cuda"):
        self.client_id    = client_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.model        = None
        self.optimizer    = None
        self.criterion    = nn.CrossEntropyLoss()
        self.dataset_size = len(train_loader.dataset)

    def set_model(self, global_model):
        self.model = copy.deepcopy(global_model).to(self.device)

    def set_optimizer(self, lr=0.001):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_local(self, epochs=5, verbose=False):
        self.model.train()
        ep_losses, ep_accs = [], []
        for ep in range(epochs):
            loss_sum, correct, total = 0.0, 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(images)
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                _, pred = out.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
            ep_losses.append(loss_sum / len(self.train_loader))
            ep_accs.append(100.0 * correct / total)
            if verbose:
                print(f"  {self.client_id} E{ep+1}: L={ep_losses[-1]:.4f} A={ep_accs[-1]:.1f}%")
        return self.model.state_dict(), ep_losses, ep_accs

    def validate(self):
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.model(images)
                loss_sum += self.criterion(out, labels).item()
                _, pred = out.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        return loss_sum / len(self.val_loader), 100.0 * correct / total

    def get_dataset_size(self):
        return self.dataset_size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEDAVG SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FedAvgServer:
    def __init__(self, global_model, clients, device="cuda"):
        self.global_model  = global_model
        self.clients       = clients
        self.device        = device
        self.total_samples = sum(c.get_dataset_size() for c in clients)
        print(f"FedAvg Server | clients={len(clients)} total={self.total_samples}")

    def aggregate_weights(self, client_weights):
        first_sd = client_weights[0][0]
        agg = {}
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
        self.global_model.eval()
        crit = nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                out = self.global_model(imgs)
                loss_sum += crit(out, labs).item()
                _, pred = out.max(1)
                total += labs.size(0)
                correct += pred.eq(labs).sum().item()
        return 100.0 * correct / total, loss_sum / len(test_loader)

    def get_global_model(self):
        return copy.deepcopy(self.global_model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QPSO SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QPSOServer:
    def __init__(self, global_model, clients, device="cuda", beta=0.7):
        self.global_model = global_model
        self.clients      = clients
        self.device       = device
        self.beta         = beta
        self.personal_best        = {}
        self.personal_best_scores = {}
        self.global_best          = None
        self.global_best_score    = 0.0
        self.mean_best            = None
        print(f"QPSO Server | clients={len(clients)} β={beta}")

    def initialize_particles(self):
        state = copy.deepcopy(self.global_model.state_dict())
        for c in self.clients:
            self.personal_best[c.client_id]        = copy.deepcopy(state)
            self.personal_best_scores[c.client_id] = 0.0
        self.global_best       = copy.deepcopy(state)
        self.global_best_score = 0.0
        print("✅ QPSO particles initialised")

    def update_personal_best(self, cid, weights, val_acc):
        if val_acc > self.personal_best_scores[cid]:
            self.personal_best[cid]        = copy.deepcopy(weights)
            self.personal_best_scores[cid] = val_acc
            return True
        return False

    def update_global_best(self, cid, val_acc):
        if val_acc > self.global_best_score:
            self.global_best       = copy.deepcopy(self.personal_best[cid])
            self.global_best_score = val_acc
            return True
        return False

    def calculate_mean_best(self):
        first_id = self.clients[0].client_id
        self.mean_best = copy.deepcopy(self.personal_best[first_id])
        for k in self.mean_best:
            self.mean_best[k] = torch.zeros_like(self.mean_best[k], dtype=torch.float32)
        for c in self.clients:
            for k in self.mean_best:
                self.mean_best[k] += self.personal_best[c.client_id][k].float()
        for k in self.mean_best:
            self.mean_best[k] /= len(self.clients)

    def qpso_aggregate(self, client_weights_list):
        for cid, w, acc in client_weights_list:
            pb = self.update_personal_best(cid, w, acc)
            gb = self.update_global_best(cid, acc)
            if pb: print(f"  {cid}: pbest ↑ {acc:.2f}%")
            if gb: print(f"  {cid}: gbest ↑ {acc:.2f}%")

        self.calculate_mean_best()
        agg = copy.deepcopy(self.global_best)

        for k in agg:
            if not agg[k].is_floating_point():
                continue

            pbest_sum = torch.zeros_like(agg[k], dtype=torch.float32)
            for c in self.clients:
                pbest_sum += self.personal_best[c.client_id][k].float()

            phi  = torch.rand_like(agg[k].float())
            u    = torch.rand_like(agg[k].float())
            p    = phi * (pbest_sum / len(self.clients)) \
                 + (1 - phi) * self.global_best[k].float()
            sign = torch.where(torch.rand_like(agg[k].float()) < 0.5,
                               torch.ones_like(agg[k].float()),
                              -torch.ones_like(agg[k].float()))

            new_val = p + sign * self.beta \
                      * torch.abs(self.mean_best[k].float() - agg[k].float()) \
                      * torch.log(1.0 / (u + 1e-8))
            agg[k] = new_val.to(self.global_best[k].dtype)

        return agg

    def evaluate_global_model(self, test_loader):
        self.global_model.eval()
        crit = nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                out = self.global_model(imgs)
                loss_sum += crit(out, labs).item()
                _, pred = out.max(1)
                total += labs.size(0)
                correct += pred.eq(labs).sum().item()
        return 100.0 * correct / total, loss_sum / len(test_loader)

    def get_global_model(self):
        return copy.deepcopy(self.global_model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEDAVG TRAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_fedavg(server, clients, global_test_loader,
                 num_rounds=100, local_epochs=5, learning_rate=0.001,
                 save_every=10, save_dir="/kaggle/working", verbose=False):
    print("=" * 80 + "\n  FEDAVG TRAINING START\n" + "=" * 80)
    history = {"round": [], "global_test_acc": [], "global_test_loss": [],
               "client1_val_acc": [], "client2_val_acc": [],
               "client3_val_acc": [], "round_time": []}
    best_acc = 0.0

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        print(f"\n{'='*60}  ROUND {rnd}/{num_rounds}")
        cw = []
        for client in clients:
            print(f"  {client.client_id} training …", end=" ")
            client.set_model(server.get_global_model())
            client.set_optimizer(learning_rate)
            w, _, _ = client.train_local(local_epochs, verbose=verbose)
            vl, va = client.validate()
            print(f"val={va:.2f}%")
            history[f"{client.client_id}_val_acc"].append(va)
            cw.append((w, client.get_dataset_size()))

        agg = server.aggregate_weights(cw)
        server.global_model.load_state_dict(agg)
        g_acc, g_loss = server.evaluate_global_model(global_test_loader)
        dt = time.time() - t0

        history["round"].append(rnd)
        history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss)
        history["round_time"].append(dt)
        print(f"  Global: acc={g_acc:.2f}% loss={g_loss:.4f} ({dt:.1f}s)")

        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/fedavg_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/fedavg_round_{rnd}.pth")
        pd.DataFrame(history).to_csv(
            f"{save_dir}/results/fedavg/metrics.csv", index=False)

    print(f"\n{'='*80}\n  FEDAVG DONE — best {best_acc:.2f}%\n{'='*80}")
    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QPSO TRAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_qpso(server, clients, global_test_loader,
               num_rounds=100, local_epochs=5, learning_rate=0.001,
               save_every=10, save_dir="/kaggle/working", verbose=False):
    print("=" * 80 + "\n  QPSO-FL TRAINING START\n" + "=" * 80)
    server.initialize_particles()
    history = {"round": [], "global_test_acc": [], "global_test_loss": [],
               "global_best_score": [],
               "client1_val_acc": [], "client2_val_acc": [], "client3_val_acc": [],
               "client1_pbest_score": [], "client2_pbest_score": [],
               "client3_pbest_score": [], "round_time": []}
    best_acc = 0.0

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        print(f"\n{'='*60}  ROUND {rnd}/{num_rounds}")
        cw = []
        for client in clients:
            print(f"  {client.client_id} training …", end=" ")
            client.set_model(server.get_global_model())
            client.set_optimizer(learning_rate)
            w, _, _ = client.train_local(local_epochs, verbose=verbose)
            vl, va = client.validate()
            print(f"val={va:.2f}%")
            history[f"{client.client_id}_val_acc"].append(va)
            cw.append((client.client_id, w, va))

        print("  QPSO aggregating …")
        agg = server.qpso_aggregate(cw)
        server.global_model.load_state_dict(agg)

        for c in clients:
            history[f"{c.client_id}_pbest_score"].append(
                server.personal_best_scores[c.client_id])
        history["global_best_score"].append(server.global_best_score)

        g_acc, g_loss = server.evaluate_global_model(global_test_loader)
        dt = time.time() - t0
        history["round"].append(rnd)
        history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss)
        history["round_time"].append(dt)
        print(f"  Global: acc={g_acc:.2f}% loss={g_loss:.4f} "
              f"gbest={server.global_best_score:.2f}% ({dt:.1f}s)")

        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/qpso_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/qpso_round_{rnd}.pth")
        pd.DataFrame(history).to_csv(
            f"{save_dir}/results/qpso/metrics.csv", index=False)

    print(f"\n{'='*80}\n  QPSO-FL DONE — best {best_acc:.2f}% "
          f"gbest {server.global_best_score:.2f}%\n{'='*80}")
    return history


print("✅ All code loaded — ready to train!")


# ─── Cell 2: Copy Preprocessed Data ──────────────────────────────────

import os, shutil

data_found = False
for search in ['/kaggle/input/fl-qpso-preprocessed-data',
               '/kaggle/input/data-preparation-fl-qpso']:
    data_src = os.path.join(search, 'data')
    if os.path.exists(data_src):
        shutil.copytree(data_src, '/kaggle/working/data', dirs_exist_ok=True)
        print(f"✅ Data copied from {data_src}")
        data_found = True
        break

if not data_found:
    # Fallback: search everywhere
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'processed' in dirs and 'test_set' in dirs:
            shutil.copytree(root, '/kaggle/working/data', dirs_exist_ok=True)
            print(f"✅ Data found at {root}")
            data_found = True
            break

if not data_found:
    print("❌ Data not found! Run this to see your inputs:")
    print("   for d in os.listdir('/kaggle/input'): print(d)")


# ─── Cell 3: Create Data Loaders ─────────────────────────────────────

data_loaders = create_data_loaders(batch_size=32, num_workers=2)


# ─── Cell 4: Quick Model Test ────────────────────────────────────────

model = create_model(num_classes=3, device=device)
x = torch.randn(2, 3, 224, 224).to(device)
print(f"Output: {model(x).shape}")  # (2, 3)
del model, x; torch.cuda.empty_cache()


# ─── Cell 5: FedAvg Training ─────────────────────────────────────────
# ⚠️ Set num_rounds=5 for a quick test first!

fedavg_model = create_model(num_classes=3, device=device)
fedavg_clients = [
    FederatedClient(f'client{i}',
                    data_loaders[f'client{i}']['train'],
                    data_loaders[f'client{i}']['val'],
                    device=device)
    for i in range(1, 4)
]
fedavg_server = FedAvgServer(fedavg_model, fedavg_clients, device=device)

fedavg_history = train_fedavg(
    server=fedavg_server,
    clients=fedavg_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=100,
    local_epochs=5,
    learning_rate=0.001,
    save_every=10,
    verbose=False,
)


# ─── Cell 6: QPSO Training ───────────────────────────────────────────
# ⚠️ Set num_rounds=5 for a quick test first!

torch.cuda.empty_cache()

qpso_model = create_model(num_classes=3, device=device)
qpso_clients = [
    FederatedClient(f'client{i}',
                    data_loaders[f'client{i}']['train'],
                    data_loaders[f'client{i}']['val'],
                    device=device)
    for i in range(1, 4)
]
qpso_server = QPSOServer(qpso_model, qpso_clients, device=device, beta=0.7)

qpso_history = train_qpso(
    server=qpso_server,
    clients=qpso_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=100,
    local_epochs=5,
    learning_rate=0.001,
    save_every=10,
    verbose=False,
)

print("\n✅ Notebook 2 complete! Save Version → Save & Run All")
