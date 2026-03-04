
import os
import glob
import shutil
import numpy as np
import nibabel as nib
import torch
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

# --- CONFIG ---
DATA_DIR = "BraTS2021_Training_Data"
DEMO_DIR = "demo_data"
NUM_SAMPLES = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "best_metric_model_refined.pth"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = "best_metric_model.pth"

# --- SETUP ---
os.makedirs(DEMO_DIR, exist_ok=True)
print(f"Preparing demo data in {DEMO_DIR}...")

# --- DATA LOADING ---
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
            data_dicts.append({"image": [t1, t1ce, t2, flair], "label": seg, "id": basename})
    return data_dicts

full_data = get_brats_file_list(DATA_DIR)
train_size = int(0.8 * len(full_data))
val_files = full_data[train_size:]
print(f"Found {len(val_files)} validation samples.")

# --- MODEL ---
print("Loading model...")
model = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# --- PROCESSING ---
print("Defining transforms...", flush=True)
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]), # Added label
    EnsureTyped(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])

print(f"Extracting {NUM_SAMPLES} samples...", flush=True)

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        sample = val_files[i]
        sample_id = sample['id']
        print(f"Processing {sample_id}...", flush=True)
        
        try:
            sample_data = transforms(sample)
            
            # Image: (4, D, H, W)
            # Label: (1, D, H, W)
            
            inputs = sample_data["image"].unsqueeze(0).to(DEVICE)
            
            print(f"  Inference on {inputs.shape}...", flush=True)
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
            outputs = (outputs.sigmoid() > 0.5).float()
            
            # Save Image (D, H, W, 4)
            img_np = inputs[0].cpu().numpy().transpose(1, 2, 3, 0)
            img_nifti = nib.Nifti1Image(img_np, affine=np.eye(4))
            nib.save(img_nifti, os.path.join(DEMO_DIR, f"{sample_id}_image.nii.gz"))
            
            # Save Label (D, H, W). Squeeze the channel (1).
            lbl_np = sample_data["label"][0].cpu().numpy()
            lbl_nifti = nib.Nifti1Image(lbl_np.astype(np.float32), affine=np.eye(4))
            nib.save(lbl_nifti, os.path.join(DEMO_DIR, f"{sample_id}_label.nii.gz"))
            
            # Save Pred (D, H, W, 3)
            pred_np = outputs[0].cpu().numpy().transpose(1, 2, 3, 0)
            pred_nifti = nib.Nifti1Image(pred_np, affine=np.eye(4))
            nib.save(pred_nifti, os.path.join(DEMO_DIR, f"{sample_id}_pred.nii.gz"))
            
        except Exception as e:
            print(f"ERROR processing {sample_id}: {e}", flush=True)
            # import traceback
            # traceback.print_exc()

print("Done!", flush=True)
