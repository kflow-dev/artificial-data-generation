import numpy as np

from src.workflows import create_vip_seed_data, impute_vip_data, compare_numeric_columns


def test_imputation_removes_nans_and_restores_schema():
    df = create_vip_seed_data(samples=50, random_state=42)
    df.loc[0, "avg_transaction"] = np.nan
    df.loc[1, "credit_score"] = np.nan

    imputed = impute_vip_data(df)

    assert imputed.isnull().sum().sum() == 0
    assert list(imputed.columns) == ["avg_transaction", "frequency", "credit_score", "vip_status"]


def test_compare_numeric_columns_returns_metrics():
    real = create_vip_seed_data(samples=30, random_state=42)
    synthetic = create_vip_seed_data(samples=30, random_state=43)
    metrics = compare_numeric_columns(real, synthetic)
    assert not metrics.empty
    assert {"column", "real_mean", "synthetic_mean", "mean_delta", "kl_divergence"}.issubset(metrics.columns)
