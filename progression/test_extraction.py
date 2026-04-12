import sys
sys.path.insert(0, 'src')

from pathlib import Path
from extract_real_trajectories import RealTrajectoryExtractor

# Setup paths
script_dir = Path("D:\Major_Project\FL_QPSO_FedAvg\progression")
raw_data_dir = script_dir / "data" / "raw" / "mu_glioma_post"
processed_data_dir = script_dir / "data" / "processed"

# Initialize and test
print("Testing trajectory extraction with 5 patients...")
extractor = RealTrajectoryExtractor(raw_data_dir, processed_data_dir)
trajectories = extractor.extract_all_trajectories(max_patients=5)

print(f"\nResults:")
for pid, traj in trajectories.items():
    print(f"  {pid}: {traj.n_timepoints} timepoints, volumes: {[f'{v:.0f}' for v in traj.volumes_mm3]}")
