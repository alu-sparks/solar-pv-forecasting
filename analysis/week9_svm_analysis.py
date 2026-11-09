from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
OUT_DIR = Path("outputs/week9_svm")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    # sklearn encodes malignant=0 and benign=1; the assignment requires malignant=1.
    y = (raw.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=3000, random_state=SEED),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [("scale", StandardScaler()), ("model", SVC(kernel="linear", C=1.0))]
        ),
        "RBF SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
            ]
        ),
    }

    scoring = {"auc": "roc_auc", "bal_acc": "balanced_accuracy"}
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_rows.append(
            {
                "Model": name,
                "ROC-AUC mean": scores["test_auc"].mean(),
                "ROC-AUC SD": scores["test_auc"].std(ddof=1),
                "Balanced accuracy mean": scores["test_bal_acc"].mean(),
                "Balanced accuracy SD": scores["test_bal_acc"].std(ddof=1),
            }
        )
    cv_results = pd.DataFrame(cv_rows)

    tuned_rbf = GridSearchCV(
        Pipeline(
            [("scale", StandardScaler()), ("model", SVC(kernel="rbf"))]
        ),
        param_grid={
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01, 0.1],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    tuned_rbf.fit(X_train, y_train)

    test_models = {
        "Logistic regression": models["Logistic regression"].fit(X_train, y_train),
        "Tuned RBF SVM": tuned_rbf.best_estimator_,
    }
    test_rows = []
    confusion = {}
    for name, model in test_models.items():
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        confusion[name] = matrix.tolist()
        test_rows.append(
            {
                "Model": name,
                "ROC-AUC": roc_auc_score(y_test, scores),
                "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
                "Malignant sensitivity": recall_score(y_test, predictions, pos_label=1),
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            }
        )
    test_results = pd.DataFrame(test_rows)

    cv_results.to_csv(OUT_DIR / "cv_results.csv", index=False)
    test_results.to_csv(OUT_DIR / "test_results.csv", index=False)

    summary = {
        "dataset_shape": [int(X.shape[0]), int(X.shape[1])],
        "class_counts": {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())},
        "train_counts": {
            "benign": int((y_train == 0).sum()),
            "malignant": int((y_train == 1).sum()),
        },
        "test_counts": {
            "benign": int((y_test == 0).sum()),
            "malignant": int((y_test == 1).sum()),
        },
        "best_params": tuned_rbf.best_params_,
        "best_cv_auc": float(tuned_rbf.best_score_),
        "confusion_matrices": confusion,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Cross-validation results")
    print(cv_results.round(4).to_string(index=False))
    print("\nBest RBF parameters:", tuned_rbf.best_params_)
    print("Best tuned CV ROC-AUC:", round(tuned_rbf.best_score_, 4))
    print("\nUntouched test results")
    print(test_results.round(4).to_string(index=False))
    print("\nClass counts:", summary)


if __name__ == "__main__":
    main()
