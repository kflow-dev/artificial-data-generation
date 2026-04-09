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
    """Recover posterior mean and scale for a selected dataframe column."""
    target_column = column or data.columns[0]
    observed = data[target_column].dropna().astype(float)
    prior_mean = float(observed.mean()) if len(observed) else 0.0
    prior_sigma = float(max(observed.std(), 1.0)) if len(observed) else 10.0

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=prior_mean, sigma=prior_sigma * 2)
        sigma = pm.HalfNormal("sigma", sigma=prior_sigma)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
        trace = pm.sample(500, tune=500, chains=2, progressbar=False)
        return trace
