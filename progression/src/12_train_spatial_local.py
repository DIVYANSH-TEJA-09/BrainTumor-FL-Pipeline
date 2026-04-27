"""
Spatial Tumor Growth Prediction - Local CPU Training
=====================================================
Adapted from the Kaggle script for local execution on CPU.
Runs ~5 hours with early stopping.

Usage:
    python src/12_train_spatial_local.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
from pathlib import Path

# ───────────────────────── Config ──────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "results" / "spatial_pairs.npz"
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

CROP_SIZE = (96, 96, 64)
BASE_FILTERS = 16
DROPOUT = 0.15
BATCH_SIZE = 2          # smaller for CPU
EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 25
DICE_WEIGHT = 0.6
BCE_WEIGHT = 0.4
FLIP_PROB = 0.5
NUM_WORKERS = 0         # 0 for Windows compatibility

device = torch.device('cpu')


# ───────────────────────── Dataset ─────────────────────────────────

class SpatialPairDataset(Dataset):
    def __init__(self, inputs, targets, augment=False):
        self.inputs = inputs
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].astype(np.float32)
        y = self.targets[idx].astype(np.float32)
        if self.augment:
            x, y = self._augment(x, y)
        return torch.from_numpy(x[np.newaxis]), torch.from_numpy(y[np.newaxis])

    def _augment(self, x, y):
        for axis in [0, 1, 2]:
            if np.random.random() < FLIP_PROB:
                x = np.flip(x, axis=axis).copy()
                y = np.flip(y, axis=axis).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k=k, axes=(0, 1)).copy()
            y = np.rot90(y, k=k, axes=(0, 1)).copy()
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0, 1)
        return x, y


# ───────────────────────── Model ───────────────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16, dropout=0.15):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.enc4 = ConvBlock3D(base * 4, base * 8, dropout)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out_conv = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        orig_shape = x.shape[2:]
        pad = []
        for s in reversed(orig_shape):
            deficit = (8 - s % 8) % 8
            pad.extend([0, deficit])
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)
        out = torch.sigmoid(self.out_conv(d1))
        out = out[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
        return out

    def _match_and_cat(self, up, skip):
        diff = [s - u for s, u in zip(skip.shape[2:], up.shape[2:])]
        if any(d != 0 for d in diff):
            up = F.pad(up, [0, diff[2], 0, diff[1], 0, diff[0]])
        return torch.cat([up, skip], dim=1)


# ───────────────────────── Loss & Metrics ──────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pf = pred.view(-1)
        tf = target.view(-1)
        inter = (pf * tf).sum()
        return 1 - (2.0 * inter + self.smooth) / (pf.sum() + tf.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.6, bce_w=0.4):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.dw = dice_w
        self.bw = bce_w

    def forward(self, pred, target):
        return self.dw * self.dice(pred, target) + self.bw * self.bce(pred, target)


def compute_dice(p, t):
    inter = (p * t).sum()
    if p.sum() + t.sum() == 0:
        return 1.0
    return float((2.0 * inter) / (p.sum() + t.sum()))


def compute_volume_error(p, t):
    vp, vt = p.sum(), t.sum()
    if vt == 0:
        return 0.0
    return float(abs(vp - vt) / vt * 100)


def compute_growth_accuracy(inp, pred, target):
    ag = np.clip(target - inp, 0, 1)
    pg = np.clip(pred - inp, 0, 1)
    if ag.sum() == 0:
        return 1.0 if pg.sum() == 0 else 0.0
    inter = (ag * pg).sum()
    return float((2.0 * inter) / (ag.sum() + pg.sum() + 1e-8))


# ───────────────────────── Training ────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_dice, n = 0, 0, 0
    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pb = (pred > 0.5).float().numpy()
        yn = y.numpy()
        total_loss += loss.item() * x.size(0)
        total_dice += compute_dice(pb, yn) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_dice / n


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_dice, total_ve, total_ga, n = 0, 0, 0, 0, 0
    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        pn = (pred > 0.5).numpy()
        yn = y.numpy()
        xn = x.numpy()
        for i in range(x.size(0)):
            total_dice += compute_dice(pn[i], yn[i])
            total_ve += compute_volume_error(pn[i], yn[i])
            total_ga += compute_growth_accuracy(xn[i], pn[i], yn[i])
            n += 1
        total_loss += loss.item() * x.size(0)
    return {
        'loss': total_loss / max(n, 1),
        'dice': total_dice / max(n, 1),
        'vol_error_pct': total_ve / max(n, 1),
        'growth_accuracy': total_ga / max(n, 1),
    }


# ───────────────────────── Main ────────────────────────────────────

def main():
    print("=" * 60)
    print("  SPATIAL TUMOR GROWTH - LOCAL CPU TRAINING")
    print("=" * 60)

    print(f"\nLoading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    inputs, targets = data['inputs'], data['targets']
    train_mask, val_mask = data['train_mask'], data['val_mask']
    print(f"  Total: {len(inputs)} | Train: {train_mask.sum()} | Val: {val_mask.sum()}")

    train_ds = SpatialPairDataset(inputs[train_mask], targets[train_mask], augment=True)
    val_ds = SpatialPairDataset(inputs[val_mask], targets[val_mask], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SpatialUNet3D(1, 1, BASE_FILTERS, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: SpatialUNet3D | Params: {n_params:,} | Device: CPU")

    criterion = CombinedLoss(DICE_WEIGHT, BCE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_dice = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [],
               'val_vol_err': [], 'val_growth_acc': []}

    print(f"\n{'Ep':>4} | {'TrLoss':>7} | {'TrDice':>7} | {'VlLoss':>7} | {'VlDice':>7} | {'VolE%':>6} | {'GrAcc':>6} | {'Time':>6}")
    print("-" * 72)

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer)
        val_m = val_epoch(model, val_loader, criterion)
        scheduler.step(val_m['dice'])

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_m['loss'])
        history['val_dice'].append(val_m['dice'])
        history['val_vol_err'].append(val_m['vol_error_pct'])
        history['val_growth_acc'].append(val_m['growth_accuracy'])

        ep_time = time.time() - ep_start
        marker = ""

        if val_m['dice'] > best_dice:
            best_dice = val_m['dice']
            patience_counter = 0
            marker = " *BEST*"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dice': best_dice,
                'config': {'base_filters': BASE_FILTERS, 'dropout': DROPOUT, 'crop_size': list(CROP_SIZE)},
            }, OUT_DIR / 'spatial_unet_best.pth')
        else:
            patience_counter += 1

        # Print every epoch (since each takes ~8 min, we want to see progress)
        print(f"{epoch:4d} | {train_loss:7.4f} | {train_dice:7.4f} | "
              f"{val_m['loss']:7.4f} | {val_m['dice']:7.4f} | "
              f"{val_m['vol_error_pct']:5.1f}% | {val_m['growth_accuracy']:6.4f} | "
              f"{ep_time/60:5.1f}m{marker}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/3600:.1f} hours. Best Dice: {best_dice:.4f}")

    # Final eval with best model
    ckpt = torch.load(OUT_DIR / 'spatial_unet_best.pth', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    final = val_epoch(model, val_loader, criterion)

    print(f"\n  Final Dice:      {final['dice']:.4f}")
    print(f"  Volume Error:    {final['vol_error_pct']:.1f}%")
    print(f"  Growth Accuracy: {final['growth_accuracy']:.4f}")

    eval_results = {
        'best_epoch': ckpt['epoch'],
        'best_dice': float(best_dice),
        'final_metrics': {k: float(v) for k, v in final.items()},
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'n_params': n_params,
        'training_time_hours': elapsed / 3600,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    with open(OUT_DIR / 'spatial_eval.json', 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  Model: results/spatial_unet_best.pth")
    print(f"  Metrics: results/spatial_eval.json")


if __name__ == "__main__":
    main()
