from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=100, verbose=2)),
    ])
