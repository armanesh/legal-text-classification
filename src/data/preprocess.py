"""
Data loading and cleaning for the Legal Text Classification dataset.
Source: https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset

Columns: Case ID, Case Outcome, Case Title, Case Text
Task: predict Case Outcome from the case text.
"""
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TARGET = "case_outcome"
TEXT_COL = "case_text"

# Outcomes with very few examples add noise without adding signal;
# grouping rare classes keeps the task meaningful for both models.
MIN_CLASS_COUNT = 100


def load_raw(filename: str = "legal_text_classification.csv") -> pd.DataFrame:
    """Load the raw legal cases CSV."""
    path = DATA_DIR / "raw" / filename
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded {len(df):,} legal cases")
    return df


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, strip excessive whitespace and citation noise."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[\d+\]", " ", text)        # remove citation markers like [1]
    text = re.sub(r"http\S+", " ", text)         # remove URLs if present
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_rare_classes(df: pd.DataFrame, min_count: int = MIN_CLASS_COUNT) -> pd.DataFrame:
    """Drop outcome classes with too few examples to learn or evaluate reliably."""
    counts = df[TARGET].value_counts()
    valid_classes = counts[counts >= min_count].index
    filtered = df[df[TARGET].isin(valid_classes)].reset_index(drop=True)
    print(f"Kept {len(valid_classes)} classes with >= {min_count} examples "
          f"({len(filtered):,} / {len(df):,} rows)")
    return filtered


def run(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = raw_df.copy()
    df = df.dropna(subset=[TEXT_COL, TARGET])
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df = df[df[TEXT_COL].str.len() > 20]  # drop near-empty texts
    df = filter_rare_classes(df)
    return df.reset_index(drop=True)
