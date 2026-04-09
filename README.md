# Artificial Data Generation

Version: v05: 2026-04-09 08:04 / v04: 2026-04-09 07:34 / v03: 2026-04-09 07:31 / v02: 2026-04-06 14:30 / v01: 2026-04-05 12:00

A notebook-first Python project in the **Data Services Portfolio** that demonstrates how to generate, corrupt, repair, analyze, and visualize synthetic customer data using classical statistics, iterative imputation, and Bayesian inference.

## Overview

This repository simulates a realistic **high-value customer / VIP analytics workflow** for a fintech-style subscription business. It combines:

- **Synthetic data generation** from multiple engines (`random`, `specialized`, `gan`)
- **Missing-data simulation and repair** using iterative imputation
- **Fidelity analysis** using KL divergence and posterior parameter recovery with PyMC
- **Visualization** of real vs. synthetic distributions
- **Jupyter notebooks** that document the end-to-end story from raw seed data to stakeholder-facing analysis

In practice, the project serves two roles:

1. a **small reusable Python package / CLI**, and
2. a **worked notebook narrative** for experimentation, demonstration, and portfolio presentation.

## Repository structure

```text
artificial-data-generation/
├── data/
│   ├── raw/                # Seed input data used in notebooks
│   ├── checkpoints/        # Intermediate datasets with corruption / enrichment
│   └── synthetic/          # Generated synthetic outputs and plots
├── notebooks/
│   ├── s00_data_pipeline.ipynb
│   ├── s01_data_generation.ipynb
│   ├── s02_data_preprocessing.ipynb
│   ├── s03_data_analysis.ipynb
│   └── s04_data_visualization.ipynb
├── src/
│   ├── analysis.py         # KL divergence + Bayesian parameter recovery
│   ├── cli.py              # Click-based command line interface
│   ├── generators.py       # Random, specialized, and GAN-style generators
│   ├── notebook_utils.py   # Notebook path helpers
│   ├── processing.py       # Missingness injection + iterative imputation
│   ├── viz.py              # Distribution comparison plots
│   └── workflows.py        # Reusable VIP data workflows
├── tests/
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Notebook workflow

The notebooks define the main business and analytical storyline.

### S00 — Data pipeline overview
Frames the business problem: a fintech company has a limited VIP customer sample and wants a trustworthy synthetic population for experimentation, analysis, and communication.

### S01 — Data generation
Builds a seed dataset and explores synthetic generation approaches:

- **Random generator**: independent draws from standard distributions
- **Specialized generator**: correlated multivariate normal generation
- **GAN-style generator**: a simplified latent transformation used as a lightweight proxy for a trained deep model

### S02 — Data preprocessing
Introduces realistic data-quality problems and shows how to repair them:

- injects missingness using workflow helpers
- imputes incomplete observations using `IterativeImputer`
- frames the task as a recoverability and data-quality exercise

### S03 — Data analysis
Evaluates synthetic fidelity through:

- **KL divergence** between reference and synthetic distributions
- **Bayesian parameter recovery** with PyMC to assess whether core statistical structure can be recovered from generated data

### S04 — Data visualization
Builds visual comparisons intended for stakeholder communication, especially around whether synthetic distributions behave like their seed counterparts.

## Python package modules

### `src/generators.py`
Defines a common `BaseGenerator` interface and three implementations:

- `RandomGenerator`
- `SpecializedGenerator`
- `GANGenerator`

All generators now emit business-facing columns:

- `avg_transaction`
- `frequency`
- `credit_score`
- `vip_status`

Use `get_generator(method)` to obtain a generator by name.

### `src/processing.py`
Provides lower-level utilities for dataset corruption and repair:

- `introduce_nans(df, proportion=0.2)`
- `bayesian_impute(df)`

### `src/workflows.py`
Provides reusable higher-level notebook and CLI workflows:

- `create_vip_seed_data(...)`
- `create_corrupted_vip_data(...)`
- `impute_vip_data(...)`
- `compare_numeric_columns(...)`
- `run_generation_workflow(...)`

### `src/analysis.py`
Contains fidelity-oriented analysis functions:

- `calculate_kl_divergence(p, q)`
- `recover_parameters(data, column=...)`

`recover_parameters` uses PyMC to estimate posterior parameters for a selected dataframe column.

### `src/viz.py`
Provides `plot_comparison(real, synthetic, filename, column=...)` for KDE-based distribution overlays.

### `src/notebook_utils.py`
Provides shared path helpers used by notebooks so they can reliably locate project data and the repository root.

### `src/cli.py`
Exposes the main command-line entry points:

- `generate`
- `bootstrap`
- `impute`
- `analyze`
- `test`

## Data assets and tracking strategy

Current repository contents include:

- `data/raw/vip_seed.csv`: seed VIP data with fields such as `avg_transaction`, `frequency`, and `credit_score`
- `data/checkpoints/vip_seed.csv`: an enriched / partially corrupted checkpoint dataset including `vip_status`
- `data/synthetic/data.csv`: a checked-in sample synthetic dataset for demonstration

### Git strategy for data

The repository now follows this practical approach:

- **keep** representative input and sample datasets under version control
- **ignore newly generated outputs** in `data/synthetic/` by default
- **avoid committing notebook checkpoints, Python caches, and build artifacts**

This keeps the project reproducible and reviewable without polluting git history with transient files.

## Installation

A dedicated Python environment is recommended.

### Option A: `pyenv` + virtualenv

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 synthetic-gen-env
pyenv local synthetic-gen-env
pip install --upgrade pip
pip install -e .
pip install -e ".[notebooks]"
```

### Option B: `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e ".[notebooks]"
```

The optional `notebooks` dependency group installs JupyterLab and related tooling for notebook execution.

## CLI usage

### Module form

Run commands as a module so relative imports inside `src` resolve correctly:

```bash
python -m src.cli --help
```

### Installed console script

After `pip install -e .`, you can also run:

```bash
artificial-data-generation --help
```

### Bootstrap notebook-ready datasets

This creates the core raw and checkpoint CSVs used by the notebooks:

```bash
python -m src.cli bootstrap \
  --samples 1000 \
  --raw-output data/raw/vip_seed.csv \
  --checkpoint-output data/checkpoints/vip_seed.csv \
  --random-state 42
```

### Generate synthetic data

```bash
python -m src.cli generate --method random --samples 1000 --output data/synthetic/random_out.csv
python -m src.cli generate --method specialized --samples 1000 --output data/synthetic/spec_out.csv
python -m src.cli generate --method gan --samples 2000 --output data/synthetic/gan_out.csv
```

### Impute missing data

```bash
python -m src.cli impute --input data/checkpoints/vip_seed.csv --output data/synthetic/imputed.csv
```

### Analyze fidelity

```bash
python -m src.cli analyze \
  --real data/raw/vip_seed.csv \
  --synth data/synthetic/gan_out.csv \
  --column avg_transaction \
  --plot-output data/synthetic/comparison.png
```

### Run tests

```bash
python -m src.cli test --suite all
pytest
```

## Running notebooks interactively

Open the notebooks in JupyterLab or VS Code and run them in sequence:

1. `s00_data_pipeline.ipynb`
2. `s01_data_generation.ipynb`
3. `s02_data_preprocessing.ipynb`
4. `s03_data_analysis.ipynb`
5. `s04_data_visualization.ipynb`

The notebooks now rely on reusable code in `src/` and shared path helpers from `src/notebook_utils.py`.

To launch JupyterLab:

```bash
jupyter lab
```

## Running each notebook from the CLI

You can execute each notebook non-interactively from the command line using Jupyter, the Makefile, or the built-in CLI helpers.

### Run notebooks in place

```bash
jupyter nbconvert --to notebook --execute notebooks/s00_data_pipeline.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s01_data_generation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s02_data_preprocessing.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s03_data_analysis.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s04_data_visualization.ipynb --inplace
```

### Run a notebook and write an executed copy elsewhere

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/s03_data_analysis.ipynb \
  --output s03_data_analysis.executed.ipynb \
  --output-dir notebooks/.executed
```

### Recommended sequence from CLI

```bash
python -m src.cli bootstrap --samples 1000 --random-state 42
jupyter nbconvert --to notebook --execute notebooks/s00_data_pipeline.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s01_data_generation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s02_data_preprocessing.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s03_data_analysis.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/s04_data_visualization.ipynb --inplace
```

### Run notebooks via the project CLI

Run one notebook in place:

```bash
python -m src.cli run-notebook notebooks/s03_data_analysis.ipynb --inplace
```

Run all notebooks into an execution folder:

```bash
python -m src.cli run-notebook --all --output-dir notebooks/.executed
```

### Run notebooks via Makefile

```bash
make bootstrap
make notebooks-run
```

Or execute them in place:

```bash
make notebooks-run-inplace
```

If you want to keep the notebooks clean in git after execution, clear outputs before committing.

## Converting notebooks to Python scripts

You can convert each notebook to a `.py` file from the CLI.

### Convert one notebook

```bash
jupyter nbconvert --to script notebooks/s01_data_generation.ipynb
```

### Convert all notebooks

```bash
jupyter nbconvert --to script notebooks/s00_data_pipeline.ipynb
jupyter nbconvert --to script notebooks/s01_data_generation.ipynb
jupyter nbconvert --to script notebooks/s02_data_preprocessing.ipynb
jupyter nbconvert --to script notebooks/s03_data_analysis.ipynb
jupyter nbconvert --to script notebooks/s04_data_visualization.ipynb
```

This will generate `.py` files alongside the notebooks.

### Convert notebooks to Python in a dedicated folder

```bash
mkdir -p notebooks/python
jupyter nbconvert --to python notebooks/s00_data_pipeline.ipynb --output-dir notebooks/python
jupyter nbconvert --to python notebooks/s01_data_generation.ipynb --output-dir notebooks/python
jupyter nbconvert --to python notebooks/s02_data_preprocessing.ipynb --output-dir notebooks/python
jupyter nbconvert --to python notebooks/s03_data_analysis.ipynb --output-dir notebooks/python
jupyter nbconvert --to python notebooks/s04_data_visualization.ipynb --output-dir notebooks/python
```

### Convert notebooks to Python via the project CLI

```bash
python -m src.cli export-notebooks --output-dir notebooks/python
```

### Convert notebooks via Makefile

```bash
make notebooks-export
```

## Strengths

- Clear end-to-end story from seed data to synthetic analysis
- Lightweight modular code under `src/`
- CLI for repeatable operations
- Reusable workflow helpers shared across notebooks and CLI
- PyMC-based Bayesian validation element
- Good separation between generation, processing, analysis, and visualization

## Current limitations and notes

- The GAN generator is a simulation placeholder, not a trained neural network.
- `calculate_kl_divergence` uses a shared-binning histogram approximation; this is practical for notebook and CLI comparisons, but not a full density-estimation framework.
- Existing tests are still relatively lightweight and could be expanded.
- Executing notebooks can regenerate data artifacts under `data/`, so review changes before committing.

## Recommended next improvements

1. Add dedicated CLI commands for notebook execution and export.
2. Persist evaluation metrics and plots to structured experiment outputs.
3. Expand tests to cover distributions, CLI behavior, and notebook-compatible schemas.
4. Add notebook kernel registration instructions for pyenv users.
5. Consider adding a richer package layout if the project grows beyond a portfolio prototype.

## Development notes

### Why `python -m src.cli`?

Using:

```bash
python -m src.cli
```

instead of:

```bash
python src/cli.py
```

is safer because it runs `src` as a package and preserves relative imports such as:

```python
from .generators import get_generator
```

## Git hygiene

The repository now includes a `.gitignore` suitable for a Python + Jupyter workflow. It covers:

- `__pycache__/`
- `*.pyc`
- `*.egg-info/`
- `.ipynb_checkpoints/`
- common virtual-environment folders
- test and coverage caches
- generated outputs in `data/synthetic/` except for the checked-in sample `data/synthetic/data.csv`

Notebook checkpoint folders, Python cache files, and egg-info metadata were also removed from git tracking.
