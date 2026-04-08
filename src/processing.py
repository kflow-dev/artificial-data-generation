import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def bayesian_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements iterative imputation which approximates Bayesian 
    regression for each missing feature.
    """
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns)

def introduce_nans(df: pd.DataFrame, proportion: float = 0.2) -> pd.DataFrame:
    """Utility to simulate data corruption."""
    df_corrupt = df.copy()
    mask = np.random.rand(*df.shape) < proportion
    df_corrupt[mask] = np.nan
    return df_corrupt
