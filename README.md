# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions, built on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset is highly imbalanced — only 0.17% of transactions are fraudulent — making this a realistic and challenging classification problem.

## Pipeline

1. **Preprocessing** — removes correlated features (Pearson > 0.85), stratified 80/20 train-test split
2. **Class Imbalance** — SMOTE applied inside the pipeline to training folds only (no data leakage)
3. **Model** — Random Forest classifier with StandardScaler
4. **Hyperparameter Tuning** — RandomizedSearchCV with StratifiedKFold (5 folds, 20 candidates, scored on ROC-AUC)
5. **Threshold Tuning** — optimal classification threshold selected by maximising F1 on the precision-recall curve
6. **Feature Importance** — post-training analysis of which features drive fraud predictions

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.978 |
| Fraud Precision | 94% |
| Fraud Recall | 79% |
| F1 (fraud class) | 0.86 |
| Optimal threshold | 0.856 |

**Confusion matrix** (56,962 test transactions):
```
                Predicted: Not Fraud    Predicted: Fraud
Actual: Not Fraud       56,859                5
Actual: Fraud               21               77
```
21 fraudulent transactions missed, 5 false alarms.

Threshold tuning (0.856 vs default 0.5) significantly reduced false alarms from 33 to 5 while keeping precision at 94%.

**Best hyperparameters found:**
- `n_estimators`: 237
- `max_depth`: 30
- `min_samples_leaf`: 10
- `min_samples_split`: 2
- `class_weight`: None (SMOTE alone sufficient)

**Top feature importances:**

| Feature | Importance |
|---|---|
| V14 | 18.2% |
| V4 | 12.4% |
| V12 | 10.6% |
| V10 | 10.6% |
| V17 | 8.2% |

V14, V4, V12, V10, and V17 account for over 60% of the model's decisions. `Time` and `Amount` ranked near the bottom, suggesting the PCA-transformed features carry most of the fraud signal.

## Folder Structure

```
├── data/
│   └── creditcard.csv     Raw dataset (gitignored)
├── notebooks/      EDA notebook
├── src/
│   ├── preprocessing.py   Data loading, redundancy removal, train-test split
│   ├── pipeline.py        Model pipeline (StandardScaler → SMOTE → RandomForest)
│   ├── train.py           Training orchestration
│   ├── tune.py            Hyperparameter tuning and threshold tuning
│   ├── evaluate.py        Metrics and feature importance
│   ├── api.py             FastAPI inference service
│   └── test_api.py        Manual API smoke tests
├── models/         Saved models (joblib)
├── Dockerfile
└── .dockerignore
```

## Usage

**Train with default settings:**
```bash
python -m src.train
```

**Train with hyperparameter tuning:**
```bash
python -m src.train --tune
```

**Evaluate a saved model with a specific threshold:**
```bash
python -m src.inference models/random_forest_pipeline_2026-03-19_22-00.joblib 0.856
```
```
Threshold: 0.8560
ROC-AUC: 0.9782
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.94      0.79      0.86        98

Confusion Matrix:
 [[56859     5]
 [   21    77]]
```

**Evaluate with auto-tuned threshold (computed from data):**
```bash
python -m src.inference models/random_forest_pipeline_2026-03-19_22-00.joblib
```

## Inference API

The FastAPI service loads the latest saved model on startup and exposes three endpoints.

**Start the server:**
```bash
python src/api.py
```

Configuration via environment variables:
- `MODELS_DIR` — directory to scan for `.joblib` files (default: `models`)
- `FRAUD_THRESHOLD` — classification threshold (default: `0.856`)

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns model filename and threshold in use |
| POST | `/predict` | Single transaction inference |
| POST | `/predict_batch` | Batch inference (max 1000 records) |

**Health check:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "random_forest_pipeline_2026-03-19_22-00.joblib", "threshold": 0.856}
```

**Single prediction:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406.0, "V1": -1.36, "V2": -0.07, ..., "Amount": 149.62}'
```
```json
{"fraud_probability": 0.012, "prediction": 0, "threshold_used": 0.856}
```

Request fields: `Time`, `V1`–`V28`, `Amount` (all `float`).

## Docker

**Build:**
```bash
docker build -t fraud-detection .
```

**Run:**
```bash
docker run -p 8000:8000 fraud-detection
```

Override defaults via environment variables:
```bash
docker run -p 8000:8000 -e FRAUD_THRESHOLD=0.9 fraud-detection
```

The image runs as a non-root user (`appuser`). The `data/` directory and notebooks are excluded from the image via `.dockerignore` — only `src/` and `models/` are copied in.

## Experiment Tracking

Runs are tracked with MLflow under the `credit-card-fraud-detection` experiment.

**Train and record a run:**
```bash
python -m src.train
python -m src.train --tune
```

**View all runs in the UI:**
```bash
mlflow ui
# open http://127.0.0.1:5000
```

Each run logs:
- **Tags:** `tuned` (true/false), `smote` (true)
- **Params:** `n_features_input`, RF hyperparameters, `threshold`
- **Metrics:** `roc_auc`, `fraud_precision`, `fraud_recall`, `fraud_f1`, `nonfraud_precision`, `nonfraud_recall`, `nonfraud_f1`, `accuracy`
- **Artifact:** the saved `.joblib` model file

MLflow run data is stored locally in `mlruns/` (gitignored).

## Requirements

- Python 3.11
- scikit-learn, imbalanced-learn, scipy
- pandas, numpy, joblib
- fastapi, uvicorn
- mlflow
