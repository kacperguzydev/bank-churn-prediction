import pandas as pd


DROP_COLUMNS = ['id', 'CustomerId', 'Surname']
CATEGORICAL_COLUMNS = ['Geography', 'Gender']


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that carry no predictive signal."""
    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=columns_to_drop)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)


def align_columns(df: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """Ensure df has exactly the same columns (and order) as training data."""
    return df.reindex(columns=reference_columns, fill_value=0)


def preprocess(df: pd.DataFrame, reference_columns: list = None) -> pd.DataFrame:
    """Full preprocessing pipeline: clean, encode, and optionally align columns."""
    df = clean_data(df)
    df = encode_features(df)
    if reference_columns is not None:
        df = align_columns(df, reference_columns)
    return df