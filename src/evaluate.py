from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score


def evaluate(pipeline, X_test, y_test, threshold=0.5):
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    print(f"Threshold: {threshold:.4f}")
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "fraud_precision": report["1"]["precision"],
        "fraud_recall": report["1"]["recall"],
        "fraud_f1": report["1"]["f1-score"],
        "nonfraud_precision": report["0"]["precision"],
        "nonfraud_recall": report["0"]["recall"],
        "nonfraud_f1": report["0"]["f1-score"],
        "accuracy": accuracy_score(y_true, y_pred),
    }


def feature_importance(pipeline, Xtest, top_n=100):
    importances = pipeline.named_steps['classifier'].feature_importances_
    features = Xtest.columns.tolist()
    pairs = list(zip(features, importances))
    #sort by importance score
    pairs.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {top_n} Feature Importances:")
    for feature, importance in pairs[:top_n]:
        print(f"  {feature}: {importance:.4f}")

