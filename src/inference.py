import joblib


def load_pipeline(path="models/random_forest_pipeline.joblib"):
    return joblib.load(path)


def predict(X, pipeline):
    return pipeline.predict(X)
