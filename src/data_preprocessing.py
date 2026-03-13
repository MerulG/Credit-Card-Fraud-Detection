import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    notebook_dir = os.path.dirname(os.path.abspath("__file__"))
    data_path = os.path.join(notebook_dir, "..", "data", "raw", "creditcard.csv")
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

def outliers(data):
    #Using IQR method per column
    for col in data.columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        data = data[(data[col]>=lower)&(data[col]<=upper)]
    return data

def test_train_split(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def balance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train

def preprocess():
    data = load_data()
    data = data_redundancy(data)
    #PCA already applied to dataset
    #data = outliers(data)
    X_train, X_test, y_train, y_test = test_train_split(data)
    X_train, X_test = scale(X_train, X_test)
    X_train, y_train = balance(X_train, y_train)
    return X_train, X_test, y_train, y_test