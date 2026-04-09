from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .analysis import bayesian_fidelity_report, calculate_kl_divergence
from .experiments import configure_tracking, log_dataframe_artifact, log_metrics, log_params
from .processing import bayesian_impute, introduce_nans
from .viz import plot_comparison, plot_correlation_heatmap


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
    cols = [c for c in [*VIP_COLUMNS, "vip_status"] if c in df.columns]
    return df[cols].describe(include="all").T


def compare_numeric_columns(real: pd.DataFrame, synthetic: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
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
    raw = create_vip_seed_data(samples=samples, random_state=random_state)
    checkpoint = create_corrupted_vip_data(raw, random_state=random_state)

    if raw_output:
        save_dataset(raw, raw_output)
    if checkpoint_output:
        save_dataset(checkpoint, checkpoint_output)

    return {"raw": raw, "checkpoint": checkpoint}


def analyze_and_log_experiment(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    output_dir: str | Path,
    tracking_uri: str | None = None,
    experiment_name: str = "artificial-data-generation",
    run_name: str | None = None,
) -> pd.DataFrame:
    """Persist metrics, comparison tables, and plots to MLflow and structured output folders."""
    mlflow = configure_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        log_params({"rows_real": len(real), "rows_synthetic": len(synthetic)})
        comparison = compare_numeric_columns(real, synthetic)
        fidelity = bayesian_fidelity_report(real, synthetic, VIP_COLUMNS)

        log_dataframe_artifact(comparison, tables_dir / "comparison_metrics.csv", artifact_path="tables")
        log_dataframe_artifact(fidelity, tables_dir / "bayesian_fidelity.csv", artifact_path="tables")

        for _, row in comparison.iterrows():
            log_metrics(
                {
                    "real_mean": row["real_mean"],
                    "synthetic_mean": row["synthetic_mean"],
                    "mean_delta": row["mean_delta"],
                    "kl_divergence": row["kl_divergence"],
                },
                prefix=str(row["column"]),
            )

        for column in VIP_COLUMNS:
            plot_comparison(real, synthetic, plots_dir / f"{column}_comparison.png", column=column)
        plot_correlation_heatmap(real[VIP_COLUMNS + ["vip_status"]], plots_dir / "real_correlation.png", "Real Correlation")
        plot_correlation_heatmap(
            synthetic[VIP_COLUMNS + ["vip_status"]],
            plots_dir / "synthetic_correlation.png",
            "Synthetic Correlation",
        )

        for artifact in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(artifact), artifact_path="plots")

    return comparison
