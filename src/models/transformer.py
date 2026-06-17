"""
Modern NLP approach: fine-tuning DistilBERT for legal case outcome classification.
Transformers capture contextual meaning beyond bag-of-words, which matters for
legal text where the same words can imply different outcomes depending on
surrounding context (e.g. negation, conditional clauses).
"""
import time
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"
MAX_LENGTH = 256


def prepare_dataset(texts: list, labels: list, label2id: dict) -> Dataset:
    """Convert text/label lists into a HuggingFace Dataset with integer labels."""
    encoded_labels = [label2id[l] for l in labels]
    return Dataset.from_dict({"text": texts, "label": encoded_labels})


def tokenize_dataset(dataset: Dataset, tokenizer):
    """Tokenize text with truncation/padding suited to legal case lengths."""
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    return dataset.map(_tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_transformer(train_df, test_df, text_col: str, target_col: str, label2id: dict, id2label: dict):
    """
    Fine-tune DistilBERT for sequence classification.
    Returns the trained model, tokenizer, and training time in seconds.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = tokenize_dataset(
        prepare_dataset(train_df[text_col].tolist(), train_df[target_col].tolist(), label2id),
        tokenizer
    )
    test_ds = tokenize_dataset(
        prepare_dataset(test_df[text_col].tolist(), test_df[target_col].tolist(), label2id),
        tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "distilbert_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start
    print(f"Fine-tuning completed in {train_time / 60:.1f} minutes")

    return trainer, train_time


def save_transformer(trainer: Trainer, tokenizer, name: str = "distilbert_legal"):
    save_path = OUTPUT_DIR / name
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Saved model to {save_path}")
    return save_path
