# GUIDE — FL QPSO vs FedAvg on Kaggle

> All code is **inline** — no source code dataset upload needed!

---

## 3 Notebooks

| # | File | Purpose | Runtime |
|---|------|---------|---------|
| 1 | `notebooks/notebook1_data_prep.py` | Preprocess 3 datasets → `.npy` | ~30 min |
| 2 | `notebooks/notebook2_training.py` | FedAvg + QPSO training | ~4-8 hrs |
| 3 | `notebooks/notebook3_evaluation.py` | Plots, stats, LaTeX tables | ~15 min |

---

## How to Use

### For Each Notebook:

1. **Create new Kaggle notebook** → GPU P100, Persistence: Files only
2. **Add datasets** via + Add Input (see below)
3. **Open the `.py` file** on your computer
4. **Copy each `# ─── Cell N ───` block** into a separate Kaggle cell
5. **Run all cells** in order
6. When done → **Save Version → Save & Run All (Commit)**

### Input Datasets

| Notebook | Add These Inputs |
|----------|-----------------|
| **1** | Brain Tumor MRI Dataset (masoudnickparvar) + BRISC 2025 (briscdataset) |
| **2** | Notebook 1's output (via + Add Input → **Notebook Output** tab → search your Notebook 1) |
| **3** | Notebook 1 output + Notebook 2 output (both via Notebook Output tab) |

### Between Notebooks

After each notebook finishes its **Save & Run All** commit:
- The output is automatically available for the next notebook
- Add it via **+ Add Input → Notebook Output tab**

---

## Dataset Paths

If Cell 3 in Notebook 1 shows ❌ for any path, run:
```python
for root, dirs, files in os.walk('/kaggle/input'):
    depth = root.replace('/kaggle/input','').count('/')
    if depth < 3: print('  '*depth + os.path.basename(root) + '/')
```
Then update `CLIENT1_PATH`, `CLIENT2_PATH`, `CLIENT3_PATH` accordingly.

---

## Tips

- **Test first**: Set `num_rounds=5` in Cells 5 & 6 of Notebook 2
- **Overnight**: Once tested, set `num_rounds=100` and commit
- **GPU check**: `!nvidia-smi`
- **If timeout**: Resume from checkpoint files in `/kaggle/working/models/`
