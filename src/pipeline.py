"""
End-to-end pipeline: load -> clean -> split -> train both approaches -> compare.
Run with: python -m src.pipeline
"""
from src.data.preprocess import load_raw, run as preprocess, TEXT_COL, TARGET
from src.data.split import split_data
from src.models.classical import build_tfidf_logreg_pipeline, train_classical, save_pipeline
from src.models.transformer import train_transformer, save_transformer
from src.evaluation.metrics import evaluate_classical, compare_models


def main():
    print("=== Legal Case Outcome Classification: Classical vs Transformer ===\n")

    # 1. Load and clean
    raw_df = load_raw()
    clean_df = preprocess(raw_df)
    class_names = sorted(clean_df[TARGET].unique())
    label2id = {label: i for i, label in enumerate(class_names)}
    id2label = {i: label for label, i in label2id.items()}

    # 2. Split (shared across both models for a fair comparison)
    train_df, test_df = split_data(clean_df)

    # 3. Classical approach: TF-IDF + Logistic Regression
    print("\n--- Training Classical Model (TF-IDF + Logistic Regression) ---")
    classical_pipeline = build_tfidf_logreg_pipeline()
    result = train_classical(classical_pipeline, train_df[TEXT_COL], train_df[TARGET])

    classical_metrics = evaluate_classical(
        result["model"], test_df[TEXT_COL], test_df[TARGET], class_names, name="TF-IDF_LogReg"
    )
    classical_metrics["train_time_seconds"] = result["train_time_seconds"]
    save_pipeline(result["model"], "tfidf_logreg")

    # 4. Transformer approach: fine-tuned DistilBERT
    print("\n--- Fine-tuning DistilBERT ---")
    trainer, train_time = train_transformer(train_df, test_df, TEXT_COL, TARGET, label2id, id2label)
    eval_results = trainer.evaluate()

    transformer_metrics = {
        "model": "DistilBERT",
        "accuracy": eval_results["eval_accuracy"],
        "f1_macro": eval_results["eval_f1_macro"],
        "train_time_seconds": train_time,
    }
    save_transformer(trainer, trainer.tokenizer if hasattr(trainer, "tokenizer") else None)

    # 5. Compare
    compare_models(classical_metrics, transformer_metrics)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
