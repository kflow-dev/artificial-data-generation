import pytest
import pandas as pd
import numpy as np
from src.processing import bayesian_impute, introduce_nans

def test_imputation_removes_nans():
    df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, np.nan, 6]})
    imputed = bayesian_impute(df)
    assert imputed.isnull().sum().sum() == 0
