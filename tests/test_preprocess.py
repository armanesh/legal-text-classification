"""Unit tests for legal text preprocessing."""
import pandas as pd
import pytest
from src.data.preprocess import clean_text, filter_rare_classes, run, TARGET, TEXT_COL


def test_clean_text_lowercases():
    assert clean_text("THE COURT RULED") == "the court ruled"


def test_clean_text_removes_citation_markers():
    result = clean_text("As stated in [1] the precedent applies")
    assert "[1]" not in result


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_filter_rare_classes_drops_small_groups():
    df = pd.DataFrame({
        TARGET: ["cited"] * 150 + ["rare_outcome"] * 5,
        TEXT_COL: ["text"] * 155
    })
    result = filter_rare_classes(df, min_count=100)
    assert "rare_outcome" not in result[TARGET].unique()
    assert "cited" in result[TARGET].unique()


def test_run_drops_short_texts():
    df = pd.DataFrame({
        TARGET: ["cited"] * 110,
        TEXT_COL: ["ok"] * 5 + ["a sufficiently long legal case description here"] * 105
    })
    result = run(df)
    assert all(len(t) > 20 for t in result[TEXT_COL])
