import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

PROC        = Path(__file__).parent.parent / "data" / "processed"
CLASS_NAMES = ["Low", "Medium", "High"]
SEP         = "=" * 65

TASKS = [
    ("accident_dataset.csv",   "accident_risk",   "Accident Risk"),
    ("congestion_dataset.csv", "congestion_risk",  "Congestion Risk"),
]

for fname, target_col, task_name in TASKS:
    print(f"\n{SEP}\nBASELINE — {task_name}\n{SEP}")

    df = pd.read_csv(PROC / fname)
    drop_cols = {"road_key", target_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target_col].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_tr, y_tr)
    y_pred = dummy.predict(X_te)

    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print(f"Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"Macro F1 : {f1_mac:.4f}")
    print()
    print(classification_report(
        y_te, y_pred, target_names=CLASS_NAMES, zero_division=0
    ))
    most_freq = CLASS_NAMES[int(y_pred[0])]
    print(
        f"*** Baseline {task_name} accuracy: {acc*100:.1f}% "
        f"(always predicts '{most_freq}') -- "
        f"any ML model must beat this to be useful ***"
    )
