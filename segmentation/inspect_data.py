
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
)
from monai.utils import set_determinism

# --- SETUP ---
print_config()
set_determinism(seed=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "BraTS2021_Training_Data" # Assuming assumes running in same dir as notebook

# --- DATA LOADING (Replicated from Notebook) ---
def get_brats_file_list(data_dir):
    train_images = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*", "*flair.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*", "*seg.nii.gz")))
    data_dicts = []
    for image_name, label_name in zip(train_images, train_labels):
        root = os.path.dirname(image_name)
        basename = os.path.basename(image_name).replace("_flair.nii.gz", "")
        t1 = os.path.join(root, f"{basename}_t1.nii.gz")
        t1ce = os.path.join(root, f"{basename}_t1ce.nii.gz")
        t2 = os.path.join(root, f"{basename}_t2.nii.gz")
        flair = os.path.join(root, f"{basename}_flair.nii.gz")
        seg = label_name
        if all(os.path.exists(p) for p in [t1, t1ce, t2, flair, seg]):
            data_dicts.append({"image": [t1, t1ce, t2, flair], "label": seg})
    return data_dicts

full_data = get_brats_file_list(data_dir)
if not full_data:
    print(f"ERROR: No data found in {data_dir}")
    exit(1)

# Use validation set (last 20%)
train_size = int(0.8 * len(full_data))
val_files = full_data[train_size:]

val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])

val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0) # workers=0 for safety

# --- MODEL ---
model = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

# Load Checkpoint
ckpt_path = "best_metric_model_refined.pth"
if not os.path.exists(ckpt_path):
    ckpt_path = "best_metric_model.pth" # Fallback
    
if os.path.exists(ckpt_path):
    print(f"Loading weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print("WARNING: No checkpoint found! Using random weights.")

# --- VISUALIZATION LOGIC ---
def visualize_deep_dive(model, val_loader):
    print("\n🔍 DEEP DIVE INSPECTION")
    print("---------------------")
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            break
            
        outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
        outputs = (outputs.sigmoid() > 0.5).float()
        
        # Dimensions
        print(f"Input Shape:  {inputs.shape} (Batch, 4 Modalities, D, H, W)")
        print(f"Output Shape: {outputs.shape} (Batch, 3 Classes, D, H, W)")
        print("\nChannels Mapping:")
        print("  Input 0: T1")
        print("  Input 1: T1ce")
        print("  Input 2: T2")
        print("  Input 3: FLAIR")
        print("  Output 0: Whole Tumor (WT)")
        print("  Output 1: Tumor Core (TC)")
        print("  Output 2: Enhancing Tumor (ET)")
        
        # Plotting
        sample_idx = 0
        slice_idx = inputs.shape[4] // 2 
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"Deep Dive Inspection (Slice {slice_idx})", fontsize=16)
        
        # Row 1: Inputs
        modality_names = ["T1", "T1ce", "T2", "FLAIR"]
        for i in range(4):
            ax = axes[0, i]
            img = inputs[sample_idx, i, :, :, slice_idx].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Input: {modality_names[i]}")
            ax.axis("off")

        # Row 2: Ground Truth
        class_names = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]
        for i in range(3):
            ax = axes[1, i]
            # Verify if label channel order matches output order (WT, TC, ET)
            # ConvertToMultiChannelBasedOnBratsClassesd does: 1->TC, 2->WT, 3->ET? 
            # Actually MONAI usually does: 
            # 0: NCR+NET (1) + ED (2) + ET (4) = WT
            # 1: NCR+NET (1) + ET (4) = TC
            # 2: ET (4)
            # So yes, indices 0, 1, 2 match WT, TC, ET.
            mask = labels[sample_idx, i, :, :, slice_idx].cpu().numpy()
            ax.imshow(mask, cmap="jet", alpha=1.0)
            ax.set_title(f"GT: {class_names[i]}")
            ax.axis("off")
            
        # GT Overlay
        ax = axes[1, 3]
        flair = inputs[sample_idx, 3, :, :, slice_idx].cpu().numpy()
        wt_gt = labels[sample_idx, 0, :, :, slice_idx].cpu().numpy()
        ax.imshow(flair, cmap="gray")
        ax.imshow(wt_gt, cmap="Greens", alpha=0.5)
        ax.set_title("GT Overlay: FLAIR + WT")
        ax.axis("off")
            
        # Row 3: Predictions
        for i in range(3):
            ax = axes[2, i]
            mask = outputs[sample_idx, i, :, :, slice_idx].cpu().numpy()
            ax.imshow(mask, cmap="jet", alpha=1.0)
            ax.set_title(f"Pred: {class_names[i]}")
            ax.axis("off")
            
        # Pred Overlay
        ax = axes[2, 3]
        flair = inputs[sample_idx, 3, :, :, slice_idx].cpu().numpy()
        wt_pred = outputs[sample_idx, 0, :, :, slice_idx].cpu().numpy()
        ax.imshow(flair, cmap="gray")
        ax.imshow(wt_pred, cmap="Reds", alpha=0.5)
        ax.set_title("Pred Overlay: FLAIR + WT")
        ax.axis("off")
        
        plt.tight_layout()
        save_path = "deep_dive_inspection.png"
        plt.savefig(save_path)
        print(f"\nSaved visualization to {save_path}")
        plt.close()

if __name__ == "__main__":
    try:
        visualize_deep_dive(model, val_loader)
    except Exception as e:
        print(f"An error occurred: {e}")
