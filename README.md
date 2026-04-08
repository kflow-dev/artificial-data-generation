# artificial data generation

Version: v02: 2026-04-06 14:30 / v01: 2026-04-05 12:00

This is one component of the Data Services Portfolio, concerning generation of data using AI/ML techniques: random data generation, data imputation, GANs.

##  CLI Usage Guide

First, install the project in editable mode:

```bash
# 1. Install the desired Python version
pyenv install 3.10.13

# 2. Create a virtual environment specifically for this project
# Syntax: pyenv virtualenv <python_version> <env_name>
pyenv virtualenv 3.10.13 synthetic-gen-env

# 3. Set the local directory to use this environment
# This creates a .python-version file in the project root
pyenv local synthetic-gen-env

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -e .
```

### 1. Generate Data
```bash
python -m src.cli generate --method gan --samples 2000 --output data/synthetic/gan_out.csv
python -m src.cli generate --method specialized --samples 1000 --output data/synthetic/spec_out.csv

### 2. Impute Missing Data
```bash
python -m src.cli impute --input data/synthetic/gan_out.csv --output data/synthetic/gan_fixed.csv
```

### 3. Bayesian Analysis
```bash
python -m src.cli analyze --real data/raw/seed.csv --synth data/synthetic/gan_out.csv
```

### 4. Run Tests
```bash
python -m src.cli test --suite all
```
```

### Note on Running:
I used `python -m src.cli` instead of `python src/cli.py`. This is the **professional standard** because it ensures that the `src` folder is treated as a package, allowing the relative imports (like `from .generators import ...`) to work perfectly without manipulating `sys.path`.
