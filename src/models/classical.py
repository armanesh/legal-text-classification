"""
Classical NLP approach: TF-IDF vectorization + Logistic Regression / Linear SVM.
Fast to train, cheap to deploy, and a strong baseline for formal, jargon-heavy
text like legal documents where vocabulary alone carries a lot of signal.
"""
import joblib
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"


def build_tfidf_logreg_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def build_tfidf_svm_pipeline() -> Pipeline:
    """TF-IDF + Linear SVM pipeline (often strong on sparse text features)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(class_weight="balanced")),
    ])


def train_classical(pipeline: Pipeline, X_train, y_train) -> dict:
    """Train a classical pipeline and report training time."""
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Trained in {train_time:.2f}s")
    return {"model": pipeline, "train_time_seconds": train_time}


def save_pipeline(pipeline: Pipeline, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    print(f"Saved pipeline to {path}")
    return path


def load_pipeline(name: str) -> Pipeline:
    return joblib.load(OUTPUT_DIR / f"{name}.joblib")
