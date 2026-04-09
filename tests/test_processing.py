import numpy as np
import pandas as pd
from src.workflows import create_vip_seed_data, impute_vip_data


def test_imputation_removes_nans_and_restores_schema():
    df = create_vip_seed_data(samples=50, random_state=42)
    df.loc[0, 'avg_transaction'] = np.nan
    df.loc[1, 'credit_score'] = np.nan

    imputed = impute_vip_data(df)

    assert imputed.isnull().sum().sum() == 0
    assert list(imputed.columns) == ['avg_transaction', 'frequency', 'credit_score', 'vip_status']
