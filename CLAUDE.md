# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commits
- Do not include co-authoring lines in commit messages.

## Running Notebooks

```bash
jupyter lab
```

## Project Architecture

This is a tabular ML classification pipeline for credit card fraud detection using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**Data flow:**
1. Raw data → `data/raw/` → processed via `src/data/` → `data/processed/`
2. Feature engineering in `src/features/`
3. Model training/evaluation in `src/models/` (Logistic Regression, RandomForest, GradientBoosting)
4. Serialized models saved to `models/`
5. FastAPI inference service (to be implemented in `src/`)
6. Docker containerization via `docker/`

**Key directories:**
- `src/data/` — data loading and preprocessing
- `src/features/` — feature engineering (scaling, encoding, missing values)
- `src/models/` — training, evaluation, and inference logic
- `src/utils/` — shared utilities
- `notebooks/` — EDA and experimentation notebooks
- `tests/` — unit and integration tests

**Evaluation metrics:** ROC-AUC, precision, recall, confusion matrix (imbalanced dataset — fraud is rare).
