from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob[:, 1]))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
