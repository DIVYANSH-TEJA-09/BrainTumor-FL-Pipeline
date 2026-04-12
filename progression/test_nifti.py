import nibabel as nib
from pathlib import Path
import numpy as np

# Test loading one segmentation file
test_file = Path("data/raw/mu_glioma_post/MU-Glioma-Post/PatientID_0003/Timepoint_1/PatientID_0003_Timepoint_1_tumorMask.nii.gz")

print(f"Testing file: {test_file.name}")
print(f"Exists: {test_file.exists()}")

if test_file.exists():
    img = nib.load(test_file)
    data = img.get_fdata()
    affine = img.affine
    
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")
    
    # Calculate voxel volume
    voxel_dims = np.abs(np.diag(affine[:3, :3]))
    voxel_volume = np.prod(voxel_dims)
    print(f"Voxel spacing: {voxel_dims}")
    print(f"Voxel volume: {voxel_volume:.4f} mm3")
    
    # Compute tumor volume
    binary_mask = (data > 0).astype(np.float32)
    tumor_voxels = np.sum(binary_mask)
    tumor_volume = tumor_voxels * voxel_volume
    print(f"Tumor voxels: {int(tumor_voxels)}")
    print(f"Tumor volume: {tumor_volume:.1f} mm3")
