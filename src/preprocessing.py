import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(notebook_dir, "..", "data", "creditcard.csv")
    return pd.read_csv(data_path)

def data_redundancy(data):
    #Using correlation matrix
    corr_matrix = data.corr().abs()
    upper = pd.DataFrame(np.triu(corr_matrix, k=1), columns=corr_matrix.columns)
    cols_to_drop = []
    for col in upper.columns:
        if any(upper[col] >= 0.85):
            cols_to_drop.append(col)
    return data.drop(cols_to_drop, axis=1)

def test_train_split(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

#note: PCA for outlieres already applied to dataset
def preprocess():
    data = load_data()
    data = data_redundancy(data)
    X_train, X_test, y_train, y_test = test_train_split(data)
    return X_train, X_test, y_train, y_test