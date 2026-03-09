# Tabular ML Classification Project - Credit-Card-Fraud-Detection

## Overview
This project demonstrates end-to-end development of a tabular machine learning classification pipeline using the Kaggle Credit Card Fraud Detection dataset. The workflow includes:

1. **Data Loading & EDA** – Basic exploratory analysis and summary statistics.
2. **Feature Engineering** – Handling missing values, scaling, and categorical encoding.
3. **Baseline & Advanced Models** – Logistic Regression, RandomForest, GradientBoosting.
4. **Evaluation** – Metrics including ROC-AUC, precision, recall, and confusion matrix.
5. **Model Deployment** – FastAPI service with input validation for predictions.
6. **Containerization** – Dockerfile to build and run the API.
7. **Cloud Deployment & Experiment Tracking** – Deploy on AWS and log experiments with MLflow.

## Folder Structure
- `data/` – Raw and processed datasets  
- `notebooks/` – Jupyter notebooks for EDA and experimentation  
- `src/` – Python modules for preprocessing, modeling, API  
- `models/` – Saved trained models (joblib/pickle)  
- `docker/` – Dockerfile and related assets  
- `README.md`  

## Requirements
- Python 3.11.7
- Pandas, NumPy, scikit-learn
- FastAPI, Pydantic
- MLflow
- Docker