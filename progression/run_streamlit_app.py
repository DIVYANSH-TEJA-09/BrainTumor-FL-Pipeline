#!/usr/bin/env python
"""
Simple launcher for Phase 2 Streamlit 3D Visualization App

Usage:
    python run_streamlit_app.py
    
This will start the Streamlit server on http://localhost:8501
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Get the progression directory
    prog_dir = Path(__file__).parent.resolve()
    app_file = prog_dir / "streamlit_3d_progression.py"
    
    if not app_file.exists():
        print(f"ERROR: App file not found: {app_file}")
        sys.exit(1)
    
    print("="*70)
    print("PHASE 2: 3D TUMOR PROGRESSION VISUALIZATION")
    print("="*70)
    print()
    print(f"Starting Streamlit app from: {app_file}")
    print(f"Working directory: {prog_dir}")
    print()
    print("The app will open at: http://localhost:8501")
    print()
    print("Features:")
    print("  - Select patient by ID or grade (HGG/LGG)")
    print("  - Navigate through timepoints with slider")
    print("  - View 3D overlay: Blue (actual) vs Red (baseline) vs Green (hybrid)")
    print("  - See volume trajectory over time")
    print("  - View per-timepoint metrics and improvement %")
    print()
    print("Press Ctrl+C to stop the server")
    print("="*70)
    print()
    
    # Verify data exists before running
    data_check = prog_dir / "streamlit_data" / "prediction_index.json"
    if not data_check.exists():
        print(f"\nERROR: Prediction data not found at {data_check}")
        print("Please run: python src/08_generate_viz_data.py")
        sys.exit(1)
    
    print(f"Data verified: {data_check} exists")
    print()
    
    # Run streamlit with explicit working directory
    import os
    os.chdir(str(prog_dir))
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file.name)]
    subprocess.run(cmd, cwd=str(prog_dir))

if __name__ == "__main__":
    main()
