"""Unit tests for the classical TF-IDF + Logistic Regression pipeline."""
import pandas as pd
from src.models.classical import build_tfidf_logreg_pipeline, train_classical


SAMPLE_TEXTS = [
    "the court found the defendant liable for damages",
    "the appeal was dismissed on procedural grounds",
    "the contract terms were upheld by the tribunal",
    "the plaintiff's claim was rejected entirely",
] * 10

SAMPLE_LABELS = ["liable", "dismissed", "upheld", "rejected"] * 10


def test_pipeline_trains_and_predicts():
    pipeline = build_tfidf_logreg_pipeline()
    result = train_classical(pipeline, SAMPLE_TEXTS, SAMPLE_LABELS)

    preds = result["model"].predict(SAMPLE_TEXTS[:4])
    assert len(preds) == 4
    assert "train_time_seconds" in result


def test_pipeline_handles_unseen_vocabulary():
    pipeline = build_tfidf_logreg_pipeline()
    pipeline.fit(SAMPLE_TEXTS, SAMPLE_LABELS)

    # Words never seen in training should not raise an error
    preds = pipeline.predict(["completely novel legal terminology never encountered before"])
    assert len(preds) == 1
