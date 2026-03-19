from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def evaluate(pipeline, X_test, y_test, threshold=0.5):
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    print(f"Threshold: {threshold:.4f}")
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def feature_importance(pipeline, Xtest, top_n=100):
    importances = pipeline.named_steps['classifier'].feature_importances_
    features = Xtest.columns.tolist()
    pairs = list(zip(features, importances))
    #sort by importance score
    pairs.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {top_n} Feature Importances:")
    for feature, importance in pairs[:top_n]:
        print(f"  {feature}: {importance:.4f}")

