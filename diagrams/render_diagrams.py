import os
import re
import subprocess
import glob

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md_file = os.path.join(base_dir, "docs", "PPT_CONTENT_AND_DIAGRAMS.md")
mermaid_dir = os.path.join(base_dir, "diagrams", "mermaid")
rendered_dir = os.path.join(base_dir, "diagrams", "rendered")

os.makedirs(mermaid_dir, exist_ok=True)
os.makedirs(rendered_dir, exist_ok=True)

# Clean up existing files
for f in glob.glob(os.path.join(mermaid_dir, "*.mmd")):
    os.remove(f)
for f in glob.glob(os.path.join(rendered_dir, "*.png")):
    os.remove(f)

print(f"Reading {md_file}...")
with open(md_file, "r", encoding="utf-8") as f:
    content = f.read()

# Find all mermaid blocks
pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
matches = pattern.findall(content)

print(f"Found {len(matches)} mermaid diagrams.")

# Descriptive names based on order in PPT_CONTENT_AND_DIAGRAMS.md
names = [
    "01_system_architecture",
    "02_data_flow",
    "03_unet_architecture",
    "04_fl_sequence",
    "05_aggregation_strategies",
    "06_experimental_setups",
    "07_progression_pipeline",
    "08_lstm_architecture",
    "09_integration_architecture",
    "10_federated_vs_centralized",
    "11_timeline_gantt"
]

mmdc_path = os.path.join(base_dir, "diagrams", "node_modules", ".bin", "mmdc.cmd")

for i, match in enumerate(matches):
    name = names[i] if i < len(names) else f"{i+1:02d}_diagram"
    mmd_file = os.path.join(mermaid_dir, f"{name}.mmd")
    png_file = os.path.join(rendered_dir, f"{name}.png")
    
    with open(mmd_file, "w", encoding="utf-8") as f:
        f.write(match.strip())
        
    print(f"Rendering {name}...")
    
    # Run local mmdc
    # -i input, -o output, -b transparent background, -t default theme
    cmd = f'"{mmdc_path}" -i "{mmd_file}" -o "{png_file}" -b transparent -t default -s 3'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error rendering {name}:")
        print(result.stderr)
    else:
        print(f"  -> Saved to {png_file}")

print("\nDone! All diagrams rendered successfully.")
