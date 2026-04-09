from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .analysis import calculate_kl_divergence
from .processing import bayesian_impute, introduce_nans


VIP_COLUMNS = ["avg_transaction", "frequency", "credit_score"]
DEFAULT_RANDOM_STATE = 42


def _rng(random_state: int | None = DEFAULT_RANDOM_STATE) -> np.random.Generator:
    return np.random.default_rng(random_state)


def add_vip_status(df: pd.DataFrame, threshold: float = 1.4) -> pd.DataFrame:
    """Derive a simple VIP label from customer activity and credit quality."""
    result = df.copy()
    frequency = result["frequency"].fillna(0.0)
    avg_transaction = result["avg_transaction"].fillna(0.0)
    credit_score = result["credit_score"].fillna(0.0)
    rfm_score = (avg_transaction / 100.0) * (credit_score / 700.0) * (frequency / 10.0)
    result["vip_status"] = (rfm_score > threshold).astype(int)
    return result


def create_vip_seed_data(samples: int = 1000, random_state: int | None = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Create a realistic seed dataset for high-value customer simulations."""
    rng = _rng(random_state)

    credit_score = np.clip(rng.normal(loc=720, scale=55, size=samples), 300, 850)
    frequency = rng.poisson(lam=12, size=samples)
    avg_transaction = np.clip(
        rng.lognormal(mean=3.6, sigma=0.45, size=samples) + 0.15 * (credit_score - 650),
        5,
        None,
    )

    seed = pd.DataFrame(
        {
            "avg_transaction": avg_transaction.round(2),
            "frequency": frequency.astype(int),
            "credit_score": credit_score.round(0),
        }
    )
    return add_vip_status(seed)


def create_corrupted_vip_data(
    df: pd.DataFrame,
    missing_proportion: float = 0.2,
    duplicate_rows: int = 30,
    random_state: int | None = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Inject realistic corruption patterns into a VIP dataset."""
    rng = _rng(random_state)
    corrupted = introduce_nans(df, proportion=missing_proportion).copy()

    if len(corrupted) > 0:
        outlier_count = min(20, len(corrupted))
        outlier_idx = rng.choice(corrupted.index.to_numpy(), size=outlier_count, replace=False)
        corrupted.loc[outlier_idx, "avg_transaction"] = (
            corrupted.loc[outlier_idx, "avg_transaction"].fillna(0) * 3.5
        ).clip(upper=5000)

        score_count = min(15, len(corrupted))
        score_idx = rng.choice(corrupted.index.to_numpy(), size=score_count, replace=False)
        corrupted.loc[score_idx, "credit_score"] = rng.choice([300, 850], size=score_count)

    if duplicate_rows > 0 and len(corrupted) > 0:
        duplicates = corrupted.sample(n=min(duplicate_rows, len(corrupted)), replace=True, random_state=random_state)
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)

    return add_vip_status(corrupted)


def impute_vip_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and restore expected business-facing columns."""
    features = df[VIP_COLUMNS].copy()
    imputed = bayesian_impute(features)
    imputed["frequency"] = imputed["frequency"].round().clip(lower=0)
    imputed["credit_score"] = imputed["credit_score"].clip(300, 850).round()
    imputed["avg_transaction"] = imputed["avg_transaction"].clip(lower=0)
    return add_vip_status(imputed)


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact descriptive summary for notebook display."""
    cols = [c for c in [*VIP_COLUMNS, "vip_status"] if c in df.columns]
    return df[cols].describe(include="all").T


def compare_numeric_columns(real: pd.DataFrame, synthetic: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Compute simple comparison metrics for shared numeric business columns."""
    if columns is None:
        columns = [c for c in VIP_COLUMNS if c in real.columns and c in synthetic.columns]

    rows = []
    for column in columns:
        real_values = real[column].dropna().to_numpy()
        synthetic_values = synthetic[column].dropna().to_numpy()
        if len(real_values) == 0 or len(synthetic_values) == 0:
            continue
        rows.append(
            {
                "column": column,
                "real_mean": float(np.mean(real_values)),
                "synthetic_mean": float(np.mean(synthetic_values)),
                "mean_delta": float(np.mean(synthetic_values) - np.mean(real_values)),
                "kl_divergence": float(calculate_kl_divergence(real_values, synthetic_values)),
            }
        )
    return pd.DataFrame(rows)


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    df.to_csv(target, index=False)
    return target


def run_generation_workflow(
    samples: int = 1000,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    raw_output: str | Path | None = None,
    checkpoint_output: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """Create seed and corrupted datasets used by the notebooks and CLI."""
    raw = create_vip_seed_data(samples=samples, random_state=random_state)
    checkpoint = create_corrupted_vip_data(raw, random_state=random_state)

    if raw_output:
        save_dataset(raw, raw_output)
    if checkpoint_output:
        save_dataset(checkpoint, checkpoint_output)

    return {"raw": raw, "checkpoint": checkpoint}
