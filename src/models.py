from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.data_preprocessing import preprocess

def train(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42, n_estimators=100, verbose=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    return y_pred, y_pred_prob

def test(y_test, y_pred, y_pred_prob):
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob[:, 1]))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def train_and_test():
    X_train, X_test, y_train, y_test = preprocess()
    y_pred, y_pred_prob = train(X_train, X_test, y_train, y_test)
    test(y_test, y_pred, y_pred_prob)

train_and_test()