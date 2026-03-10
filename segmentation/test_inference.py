import os, numpy as np, nibabel as nib, torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, Orientationd,
    Spacingd, EnsureChannelFirstd, EnsureTyped,
)

os.chdir(r"d:\Major_Project\FL_QPSO_FedAvg\segmentation")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

ckpt = "best_metric_model.pth" if os.path.exists("best_metric_model.pth") else "best_metric_model_refined.pth"
print(f"Checkpoint: {ckpt} exists={os.path.exists(ckpt)}")

model = AttentionUnet(
    spatial_dims=3, in_channels=4, out_channels=3,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
).to(DEVICE)
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()

demo = r"streamlit_app\segmentation\demo_data"
candidates = [d for d in sorted(os.listdir(demo))
               if os.path.isdir(os.path.join(demo, d)) and d.startswith("BraTS")]
print(f"Available patients: {candidates[:5]}")
pid = candidates[0]
pdir = os.path.join(demo, pid)
print(f"Testing patient: {pid}")

for m in ["t1", "t1ce", "t2", "flair", "seg"]:
    f = os.path.join(pdir, f"{pid}_{m}.nii.gz")
    print(f"  {m}: exists={os.path.exists(f)}")

data_dict = {
    "image": [
        os.path.join(pdir, f"{pid}_t1.nii.gz"),
        os.path.join(pdir, f"{pid}_t1ce.nii.gz"),
        os.path.join(pdir, f"{pid}_t2.nii.gz"),
        os.path.join(pdir, f"{pid}_flair.nii.gz"),
    ],
    "label": os.path.join(pdir, f"{pid}_seg.nii.gz"),
}

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])
sample = transforms(data_dict)
img = sample["image"]
lbl = sample["label"]
print(f"Image shape: {img.shape}")
print(f"Label shape: {lbl.shape}")
print(f"Image range: [{float(img.min()):.3f}, {float(img.max()):.3f}]")

inputs = img.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
    outputs = (outputs.sigmoid() > 0.5).float()

pred = outputs[0].cpu().numpy()
print(f"Pred shape: {pred.shape}")
print(f"TC(ch0): {pred[0].sum():.0f} voxels")
print(f"WT(ch1): {pred[1].sum():.0f} voxels")
print(f"ET(ch2): {pred[2].sum():.0f} voxels")

total_voxels = int(np.prod(pred.shape[1:]))
print(f"Total voxels: {total_voxels}")
print(f"WT coverage: {pred[1].sum()/total_voxels*100:.2f}%")

good_path = os.path.join(demo, "BraTS2021_01601_pred.nii.gz")
good = nib.load(good_path).get_fdata()
print(f"\n--- Known good (BraTS2021_01601) ---")
print(f"Shape: {good.shape}")
print(f"WT: {good[:,:,:,1].sum():.0f}, TC: {good[:,:,:,0].sum():.0f}, ET: {good[:,:,:,2].sum():.0f}")
print(f"\n--- New patient ({pid}) ---")
print(f"WT: {pred[1].sum():.0f}, TC: {pred[0].sum():.0f}, ET: {pred[2].sum():.0f}")
print("DONE")
