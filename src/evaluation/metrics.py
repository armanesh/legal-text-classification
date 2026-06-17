"""
Evaluation and side-by-side comparison of classical vs. transformer models:
accuracy, macro F1, per-class report, confusion matrix, and efficiency metrics
(training time, inference latency, model size) — the practical tradeoffs that
matter when choosing which approach to actually deploy.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def evaluate_classical(pipeline, X_test, y_test, class_names: list, name: str = "Classical") -> dict:
    """Evaluate a classical sklearn pipeline, including inference latency."""
    start = time.time()
    preds = pipeline.predict(X_test)
    inference_time = time.time() - start

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "inference_seconds_per_1000": (inference_time / len(X_test)) * 1000,
    }

    print(f"\n{name} Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  Inference time per 1000 samples: {metrics['inference_seconds_per_1000']:.3f}s")
    print("\n" + classification_report(y_test, preds, target_names=class_names))

    plot_confusion_matrix(y_test, preds, class_names, save_name=f"{name.lower()}_confusion_matrix.png")
    return metrics


def compare_models(classical_metrics: dict, transformer_metrics: dict) -> pd.DataFrame:
    """Build a side-by-side comparison table of both approaches."""
    comparison = pd.DataFrame([classical_metrics, transformer_metrics]).set_index("model")
    print("\n=== Model Comparison ===")
    print(comparison.to_string())
    return comparison


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_name: str = "confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Legal Case Outcome Classification")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / save_name, dpi=150)
    print(f"Saved confusion matrix to {FIGURES_DIR / save_name}")
    plt.close()
