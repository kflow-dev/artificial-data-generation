import pytest
import pandas as pd
from src.generators import get_generator

def test_generator_output_shape():
    gen = get_generator('random')
    df = gen.generate(100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
