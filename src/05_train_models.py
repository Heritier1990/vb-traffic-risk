import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE    = Path(__file__).parent.parent
PROC    = BASE / "data" / "processed"
FIGURES = BASE / "outputs" / "figures"
MODELS  = BASE / "models"
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEP         = "=" * 65
CLASS_NAMES = ["Low", "Medium", "High"]

def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")

_xgb_ver = tuple(int(x) for x in xgboost.__version__.split(".")[:2])
def make_xgb(**overrides) -> XGBClassifier:
    kw = dict(random_state=42, eval_metric="mlogloss", scale_pos_weight=1)
    if _xgb_ver < (2, 0):
        kw["use_label_encoder"] = False
    kw.update(overrides)
    return XGBClassifier(**kw)

# ── Parameter search grids ─────────────────────────────────────────────────
DT_GRID = {
    "max_depth":         [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
RF_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
XGB_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def run_pipeline(
    csv_path: Path,
    target_col: str,
    task_name: str,
    leaky_col: str,
    continuous_feature_names: list,
    cm_fig_name: str,
    fi_fig_name: str,
    model_pkl: str,
    scaler_pkl: str,
    features_json: str,
) -> tuple:

    section(f"TASK: {task_name}")

    # ── Load & build feature matrix ────────────────────────────────────────
    df = pd.read_csv(csv_path)
    drop_cols    = {"road_key", target_col, leaky_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values

    cont_cols = [c for c in continuous_feature_names if c in feature_cols]
    cont_idx  = [feature_cols.index(c) for c in cont_cols]

    print(f"Dataset   : {csv_path.name}  {df.shape}")
    print(f"Leaky col : '{leaky_col}' dropped")
    print(f"Features  : {feature_cols}")
    print(f"Scaled    : {cont_cols} (indices {cont_idx})")

    # ── Train / test split (80/20, stratified) ────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(y_tr):,}   Test: {len(y_te):,}")
    print(f"\n  {'Class':<10} {'Train n':>8} {'Train%':>8}  {'Test n':>8} {'Test%':>8}")
    for i, cname in enumerate(CLASS_NAMES):
        tr_n = (y_tr == i).sum(); tr_p = tr_n / len(y_tr) * 100
        te_n = (y_te == i).sum(); te_p = te_n / len(y_te) * 100
        print(f"  {cname:<10} {tr_n:>8,} {tr_p:>7.1f}%  {te_n:>8,} {te_p:>7.1f}%")

    # ── StandardScaler — fit on train, transform both ─────────────────────
    scaler = StandardScaler()
    X_tr_s = X_tr.copy()
    X_te_s = X_te.copy()
    if cont_idx:
        X_tr_s[:, cont_idx] = scaler.fit_transform(X_tr[:, cont_idx])
        X_te_s[:, cont_idx] = scaler.transform(X_te[:, cont_idx])
    print(f"\nScaler fitted on training set for: {cont_cols}")

    # ── SMOTE — training set only, after scaling ───────────────────────────
    uniq, cnts = np.unique(y_tr, return_counts=True)
    before = {CLASS_NAMES[int(u)]: int(c) for u, c in zip(uniq, cnts)}
    smt = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smt.fit_resample(X_tr_s, y_tr)
    uniq, cnts = np.unique(y_tr_sm, return_counts=True)
    after = {CLASS_NAMES[int(u)]: int(c) for u, c in zip(uniq, cnts)}
    print(f"\nSMOTE (training only):")
    print(f"  Before : {before}  total={sum(before.values()):,}")
    print(f"  After  : {after}   total={sum(after.values()):,}")

    # ── Hyperparameter tuning via RandomizedSearchCV ───────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def tune(estimator, param_grid, label, n_jobs_cv=-1):
        print(f"\nTuning {label}  (n_iter=50, 5-fold stratified CV)...")
        search = RandomizedSearchCV(
            estimator, param_distributions=param_grid,
            n_iter=50, cv=cv, scoring="f1_macro",
            random_state=42, n_jobs=n_jobs_cv, verbose=0,
        )
        search.fit(X_tr_sm, y_tr_sm)
        print(f"  Best CV macro F1 : {search.best_score_:.4f}")
        print(f"  Best params      : {search.best_params_}")
        return search.best_estimator_

    best_dt  = tune(
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        DT_GRID, "Decision Tree",
    )
    best_rf  = tune(
        # n_jobs=1 here; parallelism comes from outer RandomizedSearchCV
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        RF_GRID, "Random Forest",
    )
    best_xgb = tune(
        make_xgb(), XGB_GRID, "XGBoost",
        n_jobs_cv=1,   # XGBoost manages its own threads; avoid nested parallelism
    )

    # ── Evaluate all 4 models ─────────────────────────────────────────────
    # Baseline uses original (pre-SMOTE, pre-scale) distribution
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_tr, y_tr)

    eval_spec = {
        "Baseline (Dummy)": (dummy,    X_te),
        "Decision Tree":    (best_dt,  X_te_s),
        "Random Forest":    (best_rf,  X_te_s),
        "XGBoost":          (best_xgb, X_te_s),
    }

    results = {}
    for name, (model, X_eval) in eval_spec.items():
        y_pred = model.predict(X_eval)
        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "acc":      accuracy_score(y_te, y_pred),
            "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        }

    # Full classification report for tuned models
    for name in ["Decision Tree", "Random Forest", "XGBoost"]:
        r = results[name]
        print(f"\n{'-'*50}\n  {name}\n{'-'*50}")
        print(f"  Accuracy : {r['acc']:.4f}   Macro F1 : {r['f1_macro']:.4f}")
        print(classification_report(
            y_te, r["y_pred"], target_names=CLASS_NAMES, zero_division=0
        ))

    # Comparison table (all 4)
    print(f"\n{'Model':<22} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 44)
    for name, r in results.items():
        tag = "  <-- baseline" if name == "Baseline (Dummy)" else ""
        print(f"  {name:<20} {r['acc']:>10.4f} {r['f1_macro']:>10.4f}{tag}")

    # ── Confusion matrices: 3 tuned models side by side ───────────────────
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    ml_names = ["Decision Tree", "Random Forest", "XGBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{task_name} -- Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, ml_names):
        r  = results[name]
        cm = confusion_matrix(y_te, r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=False, linewidths=0.5,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{name}  |  Acc={r['acc']:.3f}  F1={r['f1_macro']:.3f}",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(FIGURES / cm_fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {cm_fig_name}")

    # ── Best model (highest macro F1 among tuned models) ──────────────────
    best_name  = max(ml_names, key=lambda k: results[k]["f1_macro"])
    best_r     = results[best_name]
    best_model = best_r["model"]
    baseline_lift = best_r["f1_macro"] - results["Baseline (Dummy)"]["f1_macro"]
    print(f"\nBest model : {best_name}  "
          f"(Macro F1={best_r['f1_macro']:.4f}, lift={baseline_lift:+.4f})")

    importances = best_model.feature_importances_
    fi_source   = best_name

    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(fi) * 0.45)))
    bars = ax.barh(fi.index, fi.values, color="#4e79a7")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in fi.values], padding=3, fontsize=8)
    ax.set_xlim(0, fi.max() * 1.18)
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        f"{task_name} -- Feature Importance\n({fi_source})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES / fi_fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fi_fig_name}")

    # ── Save artefacts ─────────────────────────────────────────────────────
    joblib.dump(best_model, MODELS / model_pkl)
    joblib.dump(scaler,     MODELS / scaler_pkl)
    (MODELS / features_json).write_text(json.dumps({
        "features":     feature_cols,
        "scaled_cols":  cont_cols,
        "leaky_col":    leaky_col,
        "best_model":   best_name,
    }, indent=2))
    print(f"  Saved {model_pkl}")
    print(f"  Saved {scaler_pkl}")
    print(f"  Saved {features_json}")

    return best_name, best_r["acc"], best_r["f1_macro"], results["Baseline (Dummy)"]


# ══════════════════════════════════════════════════════════════════════════
# Task 1 — Accident Risk  (drop historical_crash_rate)
# ══════════════════════════════════════════════════════════════════════════
summary = []

acc_best_name, acc_best_acc, acc_best_f1, acc_base = run_pipeline(
    csv_path                 = PROC / "accident_dataset.csv",
    target_col               = "accident_risk",
    task_name                = "Accident Risk",
    leaky_col                = "historical_crash_rate",
    continuous_feature_names = ["hour", "month", "ADT"],
    cm_fig_name              = "12_accident_model_comparison.png",
    fi_fig_name              = "14_accident_feature_importance.png",
    model_pkl                = "accident_model.pkl",
    scaler_pkl               = "accident_scaler.pkl",
    features_json            = "accident_features.json",
)
summary.append(("Accident Risk",   acc_best_name, acc_best_acc, acc_best_f1,
                acc_base["acc"],  acc_base["f1_macro"]))

# ══════════════════════════════════════════════════════════════════════════
# Task 2 — Congestion Risk  (drop ADT)
# ══════════════════════════════════════════════════════════════════════════
cong_best_name, cong_best_acc, cong_best_f1, cong_base = run_pipeline(
    csv_path                 = PROC / "congestion_dataset.csv",
    target_col               = "congestion_risk",
    task_name                = "Congestion Risk",
    leaky_col                = "ADT",
    continuous_feature_names = ["hour", "month", "historical_crash_rate"],
    cm_fig_name              = "13_congestion_model_comparison.png",
    fi_fig_name              = "15_congestion_feature_importance.png",
    model_pkl                = "congestion_model.pkl",
    scaler_pkl               = "congestion_scaler.pkl",
    features_json            = "congestion_features.json",
)
summary.append(("Congestion Risk", cong_best_name, cong_best_acc, cong_best_f1,
                cong_base["acc"], cong_base["f1_macro"]))

# ══════════════════════════════════════════════════════════════════════════
# Final cross-task summary
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")
print(f"\n{'Task':<18} {'Best Model':<18} {'Acc':>8} {'MacroF1':>9}"
      f" {'BaseAcc':>8} {'BaseF1':>8}")
print("-" * 73)
for task, model, acc, f1, b_acc, b_f1 in summary:
    print(f"  {task:<16} {model:<18} {acc:>8.4f} {f1:>9.4f}"
          f" {b_acc:>8.4f} {b_f1:>8.4f}")

print("\nSaved artefacts:")
for fname in [
    "accident_model.pkl",    "congestion_model.pkl",
    "accident_scaler.pkl",   "congestion_scaler.pkl",
    "accident_features.json","congestion_features.json",
]:
    p = MODELS / fname
    print(f"  {fname:<35} {'OK' if p.exists() else 'MISSING'}")
