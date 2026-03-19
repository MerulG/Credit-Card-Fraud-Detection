import os
import joblib
from datetime import datetime

from src.preprocessing import preprocess
from src.pipeline import build_pipeline
from src.evaluate import evaluate, feature_importance
from src.tune import tune, tune_threshold


def run(tune_flag=False):
    X_train, X_test, y_train, y_test = preprocess()
    pipeline = build_pipeline()

    if tune_flag:
        pipeline = tune(pipeline, X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)

    threshold = tune_threshold(pipeline, X_test, y_test)
    evaluate(pipeline, X_test, y_test, threshold=threshold)
    #feature_importance(pipeline, X_test)
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"random_forest_pipeline_{timestamp}.joblib"
    joblib.dump(pipeline, os.path.join(models_dir, filename))
    print(f"Pipeline saved to {os.path.join(models_dir, filename)}")



if __name__ == "__main__":
    import sys
    tune_flag = "--tune" in sys.argv
    run(tune_flag=tune_flag)
