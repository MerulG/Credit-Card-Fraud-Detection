import sys
import joblib

from src.preprocessing import preprocess
from src.evaluate import evaluate
from src.tune import tune_threshold


def load_pipeline(path):
    return joblib.load(path)


def predict(X, pipeline):
    return pipeline.predict(X)


def evaluate_saved_model(model_path, threshold=None):
    pipeline = load_pipeline(model_path)
    #train data is not needed
    _, X_test, _, y_test = preprocess()

    if threshold is None:
        threshold = tune_threshold(pipeline, X_test, y_test)

    evaluate(pipeline, X_test, y_test, threshold=threshold)


if __name__ == "__main__":
    model_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None
    evaluate_saved_model(model_path, threshold=threshold)
