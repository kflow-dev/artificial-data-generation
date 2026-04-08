import pytest
import numpy as np
from src.analysis import calculate_kl_divergence

def test_kl_divergence_same_dist():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(0, 1, 1000)
    dist = calculate_kl_divergence(a, b)
    assert dist >= 0 # KL is always non-negative
