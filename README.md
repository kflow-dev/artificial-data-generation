# Artificial Data Generation

Synthetic VIP customer data generation package with Bayesian analysis, imputation, visualization, and experiment tracking.

## What the package provides

This package focuses on reusable Python functionality for:

- generating synthetic VIP customer data
- bootstrapping seed and checkpoint datasets
- imputing missing values
- comparing real vs. synthetic data with Bayesian fidelity analysis
- producing comparison plots and correlation heatmaps
- persisting metrics, tables, and plots to structured experiment outputs with MLflow

Primary business-facing columns used across the package:

- `avg_transaction`
- `frequency`
- `credit_score`
- `vip_status`

## Package modules

### `src/generators.py`
Synthetic data generators:

- `RandomGenerator` — independent statistical baseline
- `BayesianNetworkGenerator` — posterior-inspired multivariate generator conditioned on seed data
- `GANGenerator` — neural-network GAN generator with a structured fallback when PyTorch is unavailable

Use:

```python
from src.generators import get_generator

gen = get_generator("gan")
df = gen.generate(1000, random_state=42)
```

### `src/workflows.py`
Reusable workflow functions:

- `create_vip_seed_data(...)`
- `create_corrupted_vip_data(...)`
- `impute_vip_data(...)`
- `compare_numeric_columns(...)`
- `run_generation_workflow(...)`
- `analyze_and_log_experiment(...)`

### `src/analysis.py`
Analysis utilities:

- `calculate_kl_divergence(...)`
- `recover_parameters(...)`
- `bayesian_fidelity_report(...)`

### `src/processing.py`
Lower-level preprocessing helpers:

- `introduce_nans(...)`
- `bayesian_impute(...)`

### `src/viz.py`
Plotting helpers:

- `plot_comparison(...)`
- `plot_correlation_heatmap(...)`

### `src/cli.py`
Command-line entry points:

- `generate`
- `bootstrap`
- `impute`
- `analyze`
- `run-notebook`
- `export-notebooks`
- `test`

## Installation

### Base package

```bash
pip install -e .
```

### Notebook tooling

```bash
pip install -e ".[notebooks]"
```

### GAN support

```bash
pip install -e ".[gan]"
```

## CLI examples

### Bootstrap seed and checkpoint data

```bash
python -m src.cli bootstrap \
  --samples 1000 \
  --raw-output data/raw/vip_seed.csv \
  --checkpoint-output data/checkpoints/vip_seed.csv \
  --random-state 42
```

### Generate synthetic data

```bash
python -m src.cli generate \
  --method specialized \
  --samples 1000 \
  --seed-data data/raw/vip_seed.csv \
  --output data/synthetic/spec_out.csv
```

### Generate neural GAN data

```bash
python -m src.cli generate \
  --method gan \
  --samples 1000 \
  --seed-data data/raw/vip_seed.csv \
  --epochs 200 \
  --output data/synthetic/gan_out.csv
```

### Impute missing values

```bash
python -m src.cli impute \
  --input data/checkpoints/vip_seed.csv \
  --output data/synthetic/imputed.csv
```

### Analyze and log experiments with MLflow

```bash
python -m src.cli analyze \
  --real data/raw/vip_seed.csv \
  --synth data/synthetic/gan_out.csv \
  --column avg_transaction \
  --plot-output data/synthetic/comparison.png \
  --experiment-dir experiments/latest \
  --experiment-name artificial-data-generation
```

This analysis command:

- computes KL divergence
- runs PyMC-based posterior recovery
- writes plots and tables into a structured experiment folder
- logs metrics and artifacts to MLflow

## Experiment outputs

Structured outputs are written under the selected experiment directory, for example:

```text
experiments/latest/
├── plots/
│   ├── avg_transaction_comparison.png
│   ├── frequency_comparison.png
│   ├── credit_score_comparison.png
│   ├── real_correlation.png
│   └── synthetic_correlation.png
└── tables/
    ├── comparison_metrics.csv
    └── bayesian_fidelity.csv
```

MLflow is used to persist:

- metrics
- run parameters
- comparison tables
- generated plot artifacts

## Current implementation notes

- Bayesian generation is implemented as a posterior-inspired multivariate sampler conditioned on optional seed data.
- Bayesian analysis uses PyMC for posterior recovery of selected columns.
- GAN generation uses a lightweight neural network implemented with PyTorch when available.
- If PyTorch is not installed, the GAN generator falls back to a structured simulator so the package remains usable.
- MLflow is a required dependency for experiment tracking in the current package implementation.

## Development

Run tests:

```bash
pytest
```

Run CLI help:

```bash
python -m src.cli --help
```
