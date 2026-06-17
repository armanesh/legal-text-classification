"""Stratified train/test split, shared by both modeling approaches
so the classical and transformer models are compared on identical data."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import TARGET, TEXT_COL


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified split to preserve class proportions in train and test sets."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET]
    )
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
