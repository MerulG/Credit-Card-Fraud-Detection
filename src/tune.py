from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from scipy.stats import randint

def tune(pipeline, X_train, y_train, n_iter=20, random_state=42):
    pipeline.set_params(classifier__verbose=0)

    param_grid = {
        'classifier__n_estimators': randint(50, 300),
        'classifier__max_depth': [5, 10, 20, 30],
        'classifier__min_samples_split': [2, 20],
        'classifier__min_samples_leaf': [1, 10],
        'classifier__class_weight': ['balanced', None]
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    return search.best_estimator_

def tune_threshold(pipeline, X_test, y_test):
    y_probs = pipeline.predict_proba(X_test)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores=[]
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p+r==0:
            f1_scores.append(0)
        else:
            f1_scores.append(2*p*r/(p+r))
    best_idx = f1_scores.index(max(f1_scores))
    best_threshold = thresholds[best_idx]
    print(f"Best threshold: {best_threshold:.4f}  (F1: {max(f1_scores):.4f})")
    return best_threshold

