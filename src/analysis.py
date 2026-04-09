from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import entropy


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 30) -> float:
    """Measure divergence between two one-dimensional numeric arrays.

    Arrays may have different lengths; they are compared through aligned histograms.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.size == 0 or q.size == 0:
        return float("nan")

    lower = float(min(np.min(p), np.min(q)))
    upper = float(max(np.max(p), np.max(q)))
    if np.isclose(lower, upper):
        return 0.0

    edges = np.linspace(lower, upper, bins + 1)
    p_hist, _ = np.histogram(p, bins=edges, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)

    p_hist = p_hist + 1e-9
    q_hist = q_hist + 1e-9
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(entropy(p_hist, q_hist))


def recover_parameters(data: pd.DataFrame, column: str | None = None):
    """Recover posterior mean and scale for a selected dataframe column using PyMC."""
    target_column = column or data.columns[0]
    observed = data[target_column].dropna().astype(float)
    prior_mean = float(observed.mean()) if len(observed) else 0.0
    prior_sigma = float(max(observed.std(), 1.0)) if len(observed) else 10.0

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=prior_mean, sigma=prior_sigma * 2)
        sigma = pm.HalfNormal("sigma", sigma=prior_sigma)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
        trace = pm.sample(500, tune=500, chains=2, target_accept=0.9, progressbar=False)
        return trace


def bayesian_fidelity_report(real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Combine divergence metrics with posterior mean recovery across business columns."""
    rows = []
    for column in columns:
        real_values = real[column].dropna().astype(float).to_numpy()
        synthetic_values = synthetic[column].dropna().astype(float).to_numpy()
        if len(real_values) == 0 or len(synthetic_values) == 0:
            continue
        trace = recover_parameters(synthetic[[column]], column=column)
        posterior_mean = float(trace.posterior["mu"].mean())
        rows.append(
            {
                "column": column,
                "real_mean": float(real_values.mean()),
                "synthetic_mean": float(synthetic_values.mean()),
                "posterior_mean": posterior_mean,
                "kl_divergence": calculate_kl_divergence(real_values, synthetic_values),
            }
        )
    return pd.DataFrame(rows)
