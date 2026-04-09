import pandas as pd
from src.generators import get_generator


EXPECTED_COLUMNS = {"avg_transaction", "frequency", "credit_score", "vip_status"}


def test_generator_output_shape_and_schema():
    gen = get_generator('random')
    df = gen.generate(100, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert set(df.columns) == EXPECTED_COLUMNS
