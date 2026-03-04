# Contributing to FL-QPSO Brain Tumor Management System

Thank you for your interest in contributing! This guide will help you get started.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)
- Kaggle account (for notebook execution)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/DIVYANSH-TEJA-09/BrainTumor-FL-Pipeline.git
cd FL_QPSO_FedAvg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
FL_QPSO_FedAvg/
├── segmentation/          # Module 1: 3D Attention U-Net
├── segmentation_2d/       # 2D BraTS segmentation experiments
├── federated_learning/    # Module 2: FL-QPSO classification
│   ├── src/               # Core source code
│   ├── notebooks/         # Kaggle training notebooks
│   └── setup{1,2,3}_*/    # Experimental setups
├── progression/           # Module 3: Tumor growth prediction
├── docs/                  # Documentation and guides
├── diagrams/              # Mermaid diagrams and rendered PNGs
└── presentation/          # Presentation assets
```

---

## Development Workflow

### Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, release-ready code |
| `dev` | Integration branch for active development |
| `feature/<name>` | New features (e.g., `feature/lstm-progression`) |
| `fix/<name>` | Bug fixes (e.g., `fix/qpso-convergence`) |
| `experiment/<name>` | Experimental work (e.g., `experiment/setup3-extreme-skew`) |

### Workflow

1. **Fork** the repository (external contributors) or create a branch (team members)
2. **Create a branch** from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** — commit frequently with clear messages
4. **Push** your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** against `dev`

---

## Code Style

### Python

- Follow **PEP 8** conventions
- Use **type hints** for function signatures
- Write **docstrings** for all public functions and classes
- Maximum line length: **100 characters**

```python
def train_client(
    client: FederatedClient,
    global_weights: dict,
    epochs: int = 5,
    lr: float = 0.001
) -> tuple[dict, float]:
    """
    Train a federated client locally.

    Args:
        client: The federated client instance.
        global_weights: Current global model weights.
        epochs: Number of local training epochs.
        lr: Learning rate.

    Returns:
        Tuple of (updated_weights, validation_accuracy).
    """
    ...
```

### Notebooks

- Clear all outputs before committing (reduces file size)
- Use markdown cells to explain each section
- Number cells sequentially
- Include expected runtime estimates

### Commits

Write descriptive commit messages:

```
feat(qpso): add adaptive beta scheduling for convergence

- Implement linear decay of beta from 0.9 to 0.5 over rounds
- Add beta_schedule parameter to QPSOServer
- Update trainer to pass current round to aggregation step
```

Prefixes: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `experiment`

---

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guidelines
- [ ] Docstrings are added/updated
- [ ] No model weights or large data files are committed
- [ ] Notebook outputs are cleared
- [ ] README is updated if behavior changes
- [ ] All existing code still works

### PR Description

Use the PR template provided. Include:
- What the PR does
- Which module it affects
- How to test the changes
- Screenshots/results if applicable

---

## Reporting Issues

### Bug Reports

Use the **Bug Report** issue template. Include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, GPU, OS)
- Error logs/tracebacks

### Feature Requests

Use the **Feature Request** issue template. Describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

---

## Questions?

Open a **Discussion** or reach out to the team leads:
- **Module 1 (Segmentation)** — Person 1
- **Module 2 (Classification)** — Person 1
- **Module 3 (Progression)** — Person 2 & Person 3

---

*Thank you for contributing to privacy-preserving brain tumor analysis!*
