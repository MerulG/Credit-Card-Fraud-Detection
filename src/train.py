import os
import joblib
from datetime import datetime

import mlflow

from src.preprocessing import preprocess
from src.pipeline import build_pipeline
from src.evaluate import evaluate, feature_importance, compute_metrics
from src.tune import tune, tune_threshold

# Keys searched during tuning — used to extract best params from fitted classifier
_TUNED_PARAM_KEYS = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "class_weight",
]


def run(tune_flag=False):
    X_train, X_test, y_train, y_test = preprocess()
    pipeline = build_pipeline()

    mlflow.set_experiment("credit-card-fraud-detection")

    with mlflow.start_run() as run:
        mlflow.set_tag("tuned", "true" if tune_flag else "false")
        mlflow.set_tag("smote", "true")

        mlflow.log_param("n_features_input", X_train.shape[1])

        if tune_flag:
            pipeline = tune(pipeline, X_train, y_train)
            clf_params = pipeline.named_steps["classifier"].get_params()
            for key in _TUNED_PARAM_KEYS:
                mlflow.log_param(f"classifier__{key}", clf_params[key])
        else:
            pipeline.fit(X_train, y_train)
            mlflow.log_param("classifier__n_estimators", 100)
            mlflow.log_param("classifier__random_state", 42)

        threshold = tune_threshold(pipeline, X_test, y_test)
        mlflow.log_param("threshold", threshold)

        evaluate(pipeline, X_test, y_test, threshold=threshold)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"random_forest_pipeline_{timestamp}.joblib"
        model_path = os.path.join(models_dir, filename)
        joblib.dump(pipeline, model_path)
        print(f"Pipeline saved to {model_path}")

        mlflow.log_artifact(model_path)

        run_id = run.info.run_id

    print(f"MLflow run ID: {run_id}")
    print(f"View in UI:    mlflow ui  (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    import sys
    tune_flag = "--tune" in sys.argv
    run(tune_flag=tune_flag)
