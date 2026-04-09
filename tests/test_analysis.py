import numpy as np
import pandas as pd

from src.analysis import bayesian_fidelity_report, calculate_kl_divergence


def test_kl_divergence_same_dist():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(0, 1, 1000)
    dist = calculate_kl_divergence(a, b)
    assert dist >= 0


def test_bayesian_fidelity_report_returns_expected_columns():
    real = pd.DataFrame(
        {
            "avg_transaction": [10.0, 12.0, 11.0, 13.0, 14.0],
            "frequency": [5, 6, 7, 8, 6],
            "credit_score": [650, 670, 690, 710, 730],
        }
    )
    synthetic = real.copy()
    report = bayesian_fidelity_report(real, synthetic, ["avg_transaction"])
    assert not report.empty
    assert set(report.columns) == {"column", "real_mean", "synthetic_mean", "posterior_mean", "kl_divergence"}
