import os
import numpy as np
import pandas as pd

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

def preprocess():
    data = load_data()
    data = data_redundancy(data)
    data = outliers(data)

preprocess()