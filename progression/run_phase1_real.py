"""
Simple runner for Phase 1 real trajectory extraction.
Runs in background without verbose output.
"""

from pathlib import Path
import sys
sys.path.insert(0, 'src')

from extract_real_trajectories import RealTrajectoryExtractor, run_phase1_revised

script_dir = Path(__file__).parent
raw_data_dir = script_dir / "data" / "raw" / "mu_glioma_post"
processed_data_dir = script_dir / "data" / "processed"
output_dir = script_dir / "results"
output_dir.mkdir(exist_ok=True)

print("Starting real trajectory extraction for all 203 patients...")
print("This will take 5-10 minutes. Processing in progress...\n")

# Run Phase 1 (Full dataset)
run_phase1_revised(raw_data_dir, processed_data_dir, output_dir, test_mode=False)

print("\n[DONE] Real trajectory extraction complete!")
print(f"Results saved to: {output_dir}")
