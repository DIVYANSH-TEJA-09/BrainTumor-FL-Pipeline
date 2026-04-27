# ============================================================
# CELL 0: Fix PyTorch CUDA for P100 (run FIRST, then restart runtime)
# ============================================================
import subprocess, sys
# P100 = compute 6.0, needs torch<=2.4 with cu121
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'torch==2.4.0', 'torchvision==0.19.0',
    '--index-url', 'https://download.pytorch.org/whl/cu121'])
print("Done! Now click Runtime -> Restart Runtime, then skip this cell and run Cell 1 onwards.", flush=True)

# ============================================================
# CELL 1: Setup & Find Data
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
import sys

print("PyTorch:", torch.__version__, flush=True)
print("CUDA:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "/kaggle/input/datasets/divyanshtejaedla/spatial-pairs/spatial_pairs.npz"
print(f"Data path: {DATA_PATH}", flush=True)
print(f"Exists: {os.path.exists(DATA_PATH)}", flush=True)

data = np.load(DATA_PATH)
inputs = data['inputs']
targets = data['targets']
train_mask = data['train_mask']
val_mask = data['val_mask']

print(f"Inputs:  {inputs.shape} dtype={inputs.dtype}", flush=True)
print(f"Targets: {targets.shape}", flush=True)
print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()}", flush=True)

# ============================================================
# CELL 2: Dataset Class
# ============================================================
class SpatialPairDataset(Dataset):
    def __init__(self, inputs, targets, augment=False):
        self.inputs = inputs
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].astype(np.float32)[np.newaxis]
        y = self.targets[idx].astype(np.float32)[np.newaxis]
        if self.augment:
            x, y = self._augment(x, y)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    def _augment(self, x, y):
        for axis in [1, 2, 3]:  # skip channel dim
            if np.random.random() < 0.5:
                x = np.flip(x, axis=axis)
                y = np.flip(y, axis=axis)
        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k=k, axes=(1, 2))
            y = np.rot90(y, k=k, axes=(1, 2))
        return x, y

train_ds = SpatialPairDataset(inputs[train_mask], targets[train_mask], augment=True)
val_ds = SpatialPairDataset(inputs[val_mask], targets[val_mask], augment=False)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# Quick test
x_test, y_test = train_ds[0]
print(f"Sample x: {x_test.shape}, y: {y_test.shape}", flush=True)
print("Dataset OK", flush=True)

# ============================================================
# CELL 3: 3D U-Net Model
# ============================================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        layers += [
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpatialUNet3D(nn.Module):
    def __init__(self, base=16, dropout=0.15):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.enc2 = ConvBlock3D(base, base*2)
        self.enc3 = ConvBlock3D(base*2, base*4)
        self.enc4 = ConvBlock3D(base*4, base*8, dropout)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock3D(base*8, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock3D(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base*2, base)
        self.out_conv = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        # Pad to be divisible by 8
        orig = x.shape[2:]
        pad = []
        for s in reversed(orig):
            d = (8 - s % 8) % 8
            pad.extend([0, d])
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self._cat(self.up3(e4), e3)
        d3 = self.dec3(d3)
        d2 = self._cat(self.up2(d3), e2)
        d2 = self.dec2(d2)
        d1 = self._cat(self.up1(d2), e1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.out_conv(d1))
        return out[:, :, :orig[0], :orig[1], :orig[2]]

    def _cat(self, up, skip):
        diff = [s - u for s, u in zip(skip.shape[2:], up.shape[2:])]
        if any(d != 0 for d in diff):
            up = F.pad(up, [0, diff[2], 0, diff[1], 0, diff[0]])
        return torch.cat([up, skip], dim=1)


model = SpatialUNet3D(base=16, dropout=0.15).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params: {n_params:,}", flush=True)

# Quick forward test
with torch.no_grad():
    test_out = model(x_test.unsqueeze(0).to(device))
    print(f"Forward test: {x_test.unsqueeze(0).shape} -> {test_out.shape}", flush=True)
print("Model OK", flush=True)

# ============================================================
# CELL 4: Loss & Metrics
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pf = pred.reshape(-1)
        tf = target.reshape(-1)
        inter = (pf * tf).sum()
        return 1 - (2*inter + self.smooth) / (pf.sum() + tf.sum() + self.smooth)

criterion = nn.ModuleDict({
    'dice': DiceLoss(),
    'bce': nn.BCELoss(),
}).to(device)

def combined_loss(pred, target):
    return 0.6 * criterion['dice'](pred, target) + 0.4 * criterion['bce'](pred, target)

def dice_score(pred_bin, target_bin):
    inter = (pred_bin * target_bin).sum()
    total = pred_bin.sum() + target_bin.sum()
    if total == 0: return 1.0
    return float(2 * inter / total)

def vol_error(pred_bin, target_bin):
    vp, vt = pred_bin.sum(), target_bin.sum()
    if vt == 0: return 0.0
    return float(abs(vp - vt) / vt * 100)

def growth_acc(inp, pred, target):
    ag = np.clip(target - inp, 0, 1)
    pg = np.clip(pred - inp, 0, 1)
    if ag.sum() == 0: return 1.0 if pg.sum() == 0 else 0.0
    inter = (ag * pg).sum()
    return float(2 * inter / (ag.sum() + pg.sum() + 1e-8))

print("Loss & metrics ready", flush=True)

# ============================================================
# CELL 5: TRAINING (run this - takes ~1-2 hours on GPU)
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

EPOCHS = 120
PATIENCE = 25
best_dice = 0
patience_ctr = 0
history = []

print(f"\n{'Ep':>4} | {'TrLoss':>7} | {'TrDice':>7} | {'VlDice':>7} | {'VolE%':>6} | {'GrAcc':>6} | {'Time':>5}", flush=True)
print("-" * 62, flush=True)

t_total = time.time()

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # --- Train ---
    model.train()
    tr_loss, tr_dice, tr_n = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = combined_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pb = (pred > 0.5).float()
            tr_dice += dice_score(pb.cpu().numpy(), yb.cpu().numpy()) * xb.size(0)
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
    tr_loss /= tr_n
    tr_dice /= tr_n

    # --- Validate ---
    model.eval()
    vl_dice, vl_ve, vl_ga, vl_n = 0, 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            pn = (pred > 0.5).cpu().numpy()
            yn = yb.cpu().numpy()
            xn = xb.cpu().numpy()
            for i in range(xb.size(0)):
                vl_dice += dice_score(pn[i], yn[i])
                vl_ve += vol_error(pn[i], yn[i])
                vl_ga += growth_acc(xn[i], pn[i], yn[i])
                vl_n += 1
    vl_dice /= vl_n
    vl_ve /= vl_n
    vl_ga /= vl_n

    scheduler.step(vl_dice)
    dt = time.time() - t0

    # Save best
    mark = ""
    if vl_dice > best_dice:
        best_dice = vl_dice
        patience_ctr = 0
        mark = " *"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_dice': best_dice,
            'config': {'base_filters': 16, 'dropout': 0.15, 'crop_size': [96, 96, 64]},
        }, 'spatial_unet_best.pth')
    else:
        patience_ctr += 1

    history.append({
        'epoch': epoch, 'train_loss': tr_loss, 'train_dice': tr_dice,
        'val_dice': vl_dice, 'vol_error': vl_ve, 'growth_acc': vl_ga,
    })

    # Print EVERY epoch so user sees progress
    print(f"{epoch:4d} | {tr_loss:7.4f} | {tr_dice:7.4f} | {vl_dice:7.4f} | {vl_ve:5.1f}% | {vl_ga:6.4f} | {dt:4.0f}s{mark}", flush=True)

    if patience_ctr >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}", flush=True)
        break

total_time = time.time() - t_total
print(f"\nDone in {total_time/3600:.1f} hours. Best Dice: {best_dice:.4f}", flush=True)

# ============================================================
# CELL 6: Final Evaluation & Save
# ============================================================
ckpt = torch.load('spatial_unet_best.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"\nBest model from epoch {ckpt['epoch']}", flush=True)
print(f"Best Dice: {ckpt['best_dice']:.4f}", flush=True)

# Save results
eval_results = {
    'best_epoch': ckpt['epoch'],
    'best_dice': float(best_dice),
    'n_train': int(train_mask.sum()),
    'n_val': int(val_mask.sum()),
    'n_params': n_params,
    'training_time_hours': total_time / 3600,
    'history': history,
}
with open('spatial_eval.json', 'w') as f:
    json.dump(eval_results, f, indent=2)

print("\nSaved: spatial_unet_best.pth", flush=True)
print("Saved: spatial_eval.json", flush=True)
print("\nDownload both files from the Output tab!", flush=True)
