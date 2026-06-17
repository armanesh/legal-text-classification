# Legal Case Outcome Classification: Classical vs. Transformer NLP

Predicting the outcome category of a legal case from its text, comparing a classical TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT transformer — with a side-by-side look at accuracy, training cost, and inference latency to inform which approach actually makes sense to deploy.

Legal text classification is a real enterprise NLP use case (legal tech, compliance review, case triage) that's far less common in public portfolios than sentiment analysis or news classification, while still being grounded in a genuine, well-documented dataset.

---

## Problem

Law firms and legal tech platforms process huge volumes of case text. Automatically classifying a case's likely outcome (e.g. whether a cited precedent was *applied*, *followed*, *distinguished*, or *overruled*) can speed up legal research and document triage. This project builds and compares two classifiers for that task.

**Dataset:** [Legal Text Classification Dataset — Kaggle](https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset)
25,000 legal cases with case text and annotated outcome labels.

---

## Results

| Model | Accuracy | F1 (macro) | Train Time | Inference (per 1k docs) |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~78% | ~0.71 | ~8 sec | ~0.02 sec |
| **DistilBERT (fine-tuned)** | **~85%** | **~0.79** | **~25 min** | **~4 sec** |

DistilBERT wins on accuracy and macro F1 — legal text often relies on context and clause structure (negation, conditional language) that bag-of-words features can't capture. But the classical model trains in seconds and runs orders of magnitude faster at inference, which matters for high-volume, low-latency use cases. In practice, the right choice depends on whether the bottleneck is accuracy or throughput.

---

## Project Structure

```
legal-text-classification/
├── data/
│   └── raw/                     # CSV (not committed, download from Kaggle)
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Cleaning, rare-class filtering
│   │   └── split.py             # Stratified train/test split (shared by both models)
│   ├── models/
│   │   ├── classical.py         # TF-IDF + Logistic Regression / SVM
│   │   └── transformer.py       # DistilBERT fine-tuning
│   ├── evaluation/
│   │   └── metrics.py           # Accuracy, F1, confusion matrix, efficiency comparison
│   └── pipeline.py              # End-to-end orchestration
├── notebooks/
│   └── 01_eda.ipynb             # Class distribution, text length, term frequency
├── tests/
│   ├── test_preprocess.py
│   └── test_classical.py
├── outputs/
│   ├── models/
│   └── figures/
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/armanesh/legal-text-classification.git
cd legal-text-classification
pip install -r requirements.txt
pip install -e .
```

**Get the data:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset)
2. Save the CSV as `data/raw/legal_text_classification.csv`

---

## Run the Pipeline

```bash
python -m src.pipeline
```

This will:
1. Load and clean the legal case text, filtering outcome classes with too few examples
2. Split into stratified train/test sets (shared identically across both models for a fair comparison)
3. Train TF-IDF + Logistic Regression and report training/inference time
4. Fine-tune DistilBERT for 3 epochs and evaluate
5. Print a side-by-side comparison table and save confusion matrices

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Key Design Choices

**Rare-class filtering, not arbitrary class limiting:** Legal outcome labels are long-tailed — a handful of outcomes dominate, many appear only a few times. Classes below a minimum count threshold are dropped rather than kept, since neither model can learn a reliable decision boundary from a handful of examples, and keeping them would make the evaluation metrics misleading.

**Shared stratified split:** Both models are trained and evaluated on the exact same train/test split, with class proportions preserved via stratification. This is what makes the final comparison meaningful — any accuracy difference reflects the modeling approach, not a lucky or unlucky split.

**Why measure efficiency, not just accuracy:** A model that's 7 points more accurate but 200x slower to train and run isn't automatically the right choice — it depends on the deployment constraint. Reporting train time and inference latency alongside accuracy reflects how this decision actually gets made in practice.

**DistilBERT over full BERT:** DistilBERT retains most of BERT's performance at roughly half the parameters and faster inference, which is the more realistic choice when latency and serving cost matter — exactly the kind of tradeoff this project is built to surface.

---

## Author

**Ali Rahbarimanesh** — Data Scientist & AI Engineer
[LinkedIn](https://linkedin.com/in/armanesh) · [GitHub](https://github.com/armanesh)
