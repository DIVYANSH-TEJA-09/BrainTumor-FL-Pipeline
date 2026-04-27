#!/usr/bin/env python
"""
Launcher for the Tumor Progression Forecasting Streamlit app.

Usage:
    python run_streamlit_app.py

Opens at: http://localhost:8501
"""

import subprocess
import sys
from pathlib import Path


def main():
    prog_dir = Path(__file__).parent.resolve()
    app_file = prog_dir / "streamlit_3d_progression.py"

    if not app_file.exists():
        print(f"ERROR: App file not found: {app_file}")
        sys.exit(1)

    data_check = prog_dir / "streamlit_data" / "prediction_index.json"
    if not data_check.exists():
        print(f"ERROR: Prediction data not found at {data_check}")
        print("Please run: python src/09_generate_enhanced_viz_data.py")
        sys.exit(1)

    print("=" * 60)
    print("  TUMOR PROGRESSION FORECASTING")
    print("=" * 60)
    print(f"\n  App:  {app_file.name}")
    print(f"  Data: {data_check}")
    print(f"  URL:  http://localhost:8501")
    print(f"\n  Press Ctrl+C to stop.\n")
    print("=" * 60 + "\n")

    import os
    os.chdir(str(prog_dir))
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file.name)]
    subprocess.run(cmd, cwd=str(prog_dir))


if __name__ == "__main__":
    main()
