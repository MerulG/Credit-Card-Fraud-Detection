import os
import joblib
from datetime import datetime

from src.preprocessing import preprocess
from src.pipeline import build_pipeline
from src.evaluate import evaluate


def run():
    X_train, X_test, y_train, y_test = preprocess()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    evaluate(pipeline, X_test, y_test)
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"random_forest_pipeline_{timestamp}.joblib"
    joblib.dump(pipeline, os.path.join(models_dir, filename))
    print(f"Pipeline saved to {os.path.join(models_dir, filename)}")


if __name__ == "__main__":
    run()
