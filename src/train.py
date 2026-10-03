"""Reproducible training pipeline.

Methodology (see README for the full rationale):

1. Load + clean the dataset with an auditable cleaning report.
2. Split train/test **stratified** *before* anything is fitted.
3. Fit all preprocessing **inside** an sklearn ``Pipeline`` on the training
   rows only - scaling, imputation and encoding therefore never see the
   test set.
4. Select the model and its hyperparameters by cross-validation on the
   **training set only**, scored with threshold-independent ROC-AUC.
5. Derive the single authoritative decision threshold from out-of-fold
   predictions on the training set (test set still untouched).
6. Refit on the full training set and evaluate **once** on the held-out
   test set.

Run with::

    python -m src.train
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from .data import file_sha256, load_and_clean

# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------

SEED = cfg.RANDOM_STATE
CV = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=SEED)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cfg.PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------

def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + estimator, fitted only on whatever it is trained on."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), cfg.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             cfg.CATEGORICAL_FEATURES),
            ("bin", "passthrough", cfg.BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def candidate_models() -> dict[str, tuple[dict, dict]]:
    """``name -> (base estimator, parameter grid)``.

    Grids are deliberately modest: the point is a *fair* comparison under one
    shared cross-validation protocol, not an exhaustive search.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(
                solver="liblinear", max_iter=5000, random_state=SEED,
            ),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "model__n_estimators": [200],
                "model__max_depth": [None, 12],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {
                "model__n_estimators": [200, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3],
            },
        ),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int | None = None
) -> float:
    """Expected calibration error over equal-width probability bins.

    0.0 means that among all rows the model scores ~0.7, about 70% really are
    positive. Lower is better; it is insensitive to ranking, so it measures
    the *scale* of the probabilities rather than their ordering.
    """
    if n_bins is None:
        n_bins = cfg.CALIBRATION_BINS
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(y_proba, edges) - 1, 0, n_bins - 1)
    total = len(y_proba)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(
            y_true[mask].mean() - y_proba[mask].mean()
        )
    return round(float(ece), 4)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Full metric set. ``y_proba`` drives the threshold-free metrics.

    Discrimination (ROC-AUC / PR-AUC) and calibration (Brier / ECE) are both
    reported: a model can rank perfectly and still hand out badly scaled
    probabilities, and only the second one is safe to show to a user.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "npv": round(float(npv), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "ece": expected_calibration_error(y_true, y_proba),
        "confusion_matrix": {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        },
    }


def held_out_split(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the exact test split used during training.

    Deterministic (same seed, same stratification), so a report can score the
    held-out rows without refitting or re-tuning anything.
    """
    if df is None:
        df, _ = load_and_clean()
    X = df[cfg.MODEL_FEATURES].copy()
    y = df[cfg.TARGET].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=SEED, stratify=y
    )
    return X_test, y_test


def subgroup_metrics(
    pipeline: Pipeline,
    threshold: float,
    df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Read-only fairness check over the held-out test set.

    Scores the *already fitted* pipeline at the production threshold and
    reports per-group metrics. Nothing is fitted here, so this can never
    change the model - it only measures it.

    Groups with fewer than ``cfg.SUBGROUP_MIN_N`` rows, or containing a single
    class, get ``n``/``prevalence`` and NaN metrics instead of a noisy number.
    """
    if df is None:
        df, _ = load_and_clean()
    X_test, y_test = held_out_split(df)

    proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=X_test.index)
    pred = pd.Series((proba.values >= threshold).astype(int), index=X_test.index)

    groups = pd.DataFrame(index=X_test.index)
    for col in cfg.SUBGROUP_COLUMNS:
        groups[col] = df.loc[X_test.index, col]
    groups["age_band"] = pd.cut(
        df.loc[X_test.index, "Age"],
        cfg.AGE_BAND_EDGES,
        labels=cfg.AGE_BAND_LABELS,
    )

    metric_keys = (
        "predicted_positive_rate",
        "precision",
        "recall",
        "f1",
        "specificity",
        "accuracy",
    )

    report: dict[str, pd.DataFrame] = {}
    for col in groups.columns:
        rows: list[dict[str, Any]] = []
        for value, idx in groups.groupby(col, observed=True).groups.items():
            yt = y_test.loc[idx]
            row: dict[str, Any] = {
                "group": str(value),
                "n": int(len(idx)),
                "prevalence": round(float(yt.mean()), 4),
            }
            if len(idx) < cfg.SUBGROUP_MIN_N or yt.nunique() < 2:
                row.update({k: np.nan for k in metric_keys})
                row["note"] = "too small / single class"
            else:
                m = compute_metrics(
                    yt.values, pred.loc[idx].values, proba.loc[idx].values
                )
                row.update(
                    {
                        "predicted_positive_rate": round(
                            float(pred.loc[idx].mean()), 4
                        ),
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "specificity": m["specificity"],
                        "accuracy": m["accuracy"],
                    }
                )
            rows.append(row)
        report[col] = pd.DataFrame(rows).set_index("group")
    return report


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = cfg.THRESHOLD_OBJECTIVE,
) -> dict[str, Any]:
    """Pick the single authoritative decision threshold.

    Search is on a 0.01 grid over [0.01, 0.99]. Among all thresholds that
    attain the maximum objective value, the one **closest to 0.5** is chosen
    so the result is stable rather than pinned to whichever end of a plateau
    ``argmax`` happened to hit first.
    """
    grid = np.linspace(0.01, 0.99, 99)
    scores = np.array([
        f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        for t in grid
    ]) if objective == "f1" else np.array([
        recall_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        for t in grid
    ])

    best_value = float(scores.max())
    tied = grid[np.isclose(scores, best_value)]
    best_threshold = float(tied[np.argmin(np.abs(tied - 0.5))])

    return {
        "threshold": round(best_threshold, 2),
        "objective": objective,
        "objective_value": round(best_value, 4),
        "selection_data": "out-of-fold predictions on the training set",
        "grid": "0.01 steps over [0.01, 0.99]",
        "tie_break": "among tied thresholds, the one closest to 0.5",
    }


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------

def select_candidate(
    candidates: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any]]:
    """Decide which candidate configuration to fit.

    ``candidates`` must have ``model``, ``params``, ``cv_roc_auc_mean``,
    ``cv_roc_auc_std`` and ``cv_brier`` (out-of-fold Brier, lower is better).

    The rule, in order:

    1. Find the best mean CV ROC-AUC.
    2. Keep every candidate within **one standard error** of it
       (``std / sqrt(n_folds)``) - i.e. every candidate that is
       indistinguishable from the winner on ranking quality.
    3. Among those, take the **lowest out-of-fold Brier score**: the best
       calibrated model among the ones that rank equally well.

    Without step 3 a 0.0001 AUC difference - far below the +/-0.002
    fold-to-fold noise - decides the model, and it happily picks a
    configuration whose probabilities are visibly off-scale. Step 2 stops
    that difference from mattering; step 3 then rewards honest probabilities.

    Ties break toward higher ROC-AUC, then alphabetical model name, so the
    outcome is deterministic. Returns the chosen row plus an audit record of
    the decision.
    """
    if candidates.empty:
        raise ValueError("no candidates to select from")

    best_auc = float(candidates["cv_roc_auc_mean"].max())
    best_row = candidates.loc[candidates["cv_roc_auc_mean"].idxmax()]
    se = float(best_row["cv_roc_auc_std"]) / np.sqrt(cfg.CV_FOLDS)
    floor = best_auc - se

    eligible = candidates[candidates["cv_roc_auc_mean"] >= floor]
    chosen = eligible.sort_values(
        by=["cv_brier", "cv_roc_auc_mean", "model"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]

    info: dict[str, Any] = {
        "rule": "one standard error on CV ROC-AUC, then lowest out-of-fold Brier",
        "tie_break": "lowest Brier, then highest ROC-AUC, then model name",
        "n_candidates": int(len(candidates)),
        "best_cv_roc_auc": round(best_auc, 4),
        "one_standard_error": round(se, 6),
        "eligibility_floor": round(floor, 4),
        "n_eligible": int(len(eligible)),
        "selected_model": str(chosen["model"]),
        "selected_params": _jsonify(dict(chosen["params"])),
        "selected_cv_roc_auc": round(float(chosen["cv_roc_auc_mean"]), 4),
        "selected_cv_brier": round(float(chosen["cv_brier"]), 4),
    }
    return chosen, info


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def run_training(save: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Execute the full pipeline and return a structured result."""

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    np.random.seed(SEED)

    # ---- 1. Data --------------------------------------------------------
    log("=" * 72)
    log("STEP 1  Load and clean data")
    log("=" * 72)
    raw = pd.read_csv(cfg.DATA_PATH)
    df, cleaning_report = load_and_clean()
    log(cleaning_report.summary())
    log()

    X = df[cfg.MODEL_FEATURES].copy()
    y = df[cfg.TARGET].astype(int)
    log(f"Model features ({len(cfg.MODEL_FEATURES)}): {cfg.MODEL_FEATURES}")
    log(f"Class distribution: {dict(y.value_counts().sort_index())} "
        f"(positive rate {y.mean():.4f})")
    log()

    # ---- 2. Split BEFORE anything is fitted ----------------------------
    log("=" * 72)
    log("STEP 2  Stratified train/test split (split before fitting)")
    log("=" * 72)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    log(f"train={len(X_train)}  test={len(X_test)}")
    log(f"train positive rate={y_train.mean():.4f}  "
        f"test positive rate={y_test.mean():.4f}")
    log("Preprocessing is fit inside the Pipeline on training rows only,")
    log("so scaling/imputation/encoding never observe the test set.")
    log()

    # ---- 3. Model selection by CV on TRAIN ------------------------------
    log("=" * 72)
    log(f"STEP 3  Model selection by {cfg.CV_FOLDS}-fold CV on the training "
        f"set (scoring={cfg.SELECTION_SCORING})")
    log("=" * 72)

    comparison: dict[str, Any] = {}
    searches: dict[str, GridSearchCV] = {}
    rows: list[dict[str, Any]] = []
    for name, (estimator, grid) in candidate_models().items():
        search = GridSearchCV(
            build_pipeline(estimator),
            param_grid=grid,
            # Two metrics on the same fitted folds: AUC for ranking quality,
            # Brier for how honest the probabilities are. Costs no extra fits.
            scoring={
                "roc_auc": cfg.SELECTION_SCORING,
                "brier": "neg_brier_score",
            },
            cv=CV,
            n_jobs=-1,
            refit="roc_auc",
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        searches[name] = search

        results = search.cv_results_
        for i, params in enumerate(results["params"]):
            rows.append({
                "model": name,
                "params": params,
                "cv_roc_auc_mean": round(float(results["mean_test_roc_auc"][i]), 4),
                "cv_roc_auc_std": round(float(results["std_test_roc_auc"][i]), 4),
                "cv_brier": round(-float(results["mean_test_brier"][i]), 4),
            })

        best = search.best_estimator_
        # OOF metrics for this model's own AUC-best configuration.
        cv_proba = cross_val_predict(
            best, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1,
        )[:, 1]
        cv_pred = (cv_proba >= 0.5).astype(int)
        cv_metrics = compute_metrics(y_train.to_numpy(), cv_pred, cv_proba)

        comparison[name] = {
            "best_params": {
                k.replace("model__", ""): _jsonify(v)
                for k, v in search.best_params_.items()
            },
            "cv_roc_auc_mean": round(float(search.best_score_), 4),
            "cv_roc_auc_std": round(float(results["std_test_roc_auc"][
                search.best_index_]), 4),
            "cv_brier": cv_metrics["brier"],
            "cv_ece": cv_metrics["ece"],
            "n_candidates": len(results["params"]),
            "cv_metrics_at_0.5": cv_metrics,
            "train_accuracy": round(float(best.score(X_train, y_train)), 4),
        }
        log(f"\n{name}")
        log(f"  best params      : {comparison[name]['best_params']}")
        log(f"  CV ROC-AUC       : {comparison[name]['cv_roc_auc_mean']:.4f} "
            f"+/- {comparison[name]['cv_roc_auc_std']:.4f} "
            f"({comparison[name]['n_candidates']} candidates)")
        log(f"  CV Brier / ECE   : {comparison[name]['cv_brier']:.4f} / "
            f"{comparison[name]['cv_ece']:.4f}")
        log(f"  CV metrics @0.5  : acc={cv_metrics['accuracy']} "
            f"prec={cv_metrics['precision']} rec={cv_metrics['recall']} "
            f"f1={cv_metrics['f1']} spec={cv_metrics['specificity']} "
            f"pr_auc={cv_metrics['pr_auc']}")

    # Selection: rank on AUC, then break ties on calibration (train only).
    chosen_row, selection = select_candidate(pd.DataFrame(rows))
    selected_name = selection["selected_model"]
    best_pipeline = build_pipeline(
        candidate_models()[selected_name][0]
    ).set_params(**chosen_row["params"])
    log()
    log(f">>> SELECTED: {selected_name}")
    log(f"    rule: {selection['rule']}")
    log(f"    {selection['n_candidates']} candidates, "
        f"{selection['n_eligible']} within one standard error "
        f"(floor {selection['eligibility_floor']:.4f} = best "
        f"{selection['best_cv_roc_auc']:.4f} - "
        f"{selection['one_standard_error']:.4f})")
    log(f"    chosen by lowest OOF Brier "
        f"({selection['selected_cv_brier']:.4f}) "
        f"at CV ROC-AUC {selection['selected_cv_roc_auc']:.4f}")
    log("    The held-out test set has not been used for this decision.")
    log()

    # ---- 4. Threshold from out-of-fold TRAIN predictions ----------------
    log("=" * 72)
    log("STEP 4  Authoritative decision threshold")
    log("=" * 72)
    oof_proba = cross_val_predict(
        best_pipeline, X_train, y_train, cv=CV,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    threshold_info = select_threshold(y_train.to_numpy(), oof_proba)
    threshold = threshold_info["threshold"]
    log(f"Selected threshold = {threshold} "
        f"({cfg.THRESHOLD_OBJECTIVE}={threshold_info['objective_value']} on "
        f"out-of-fold training predictions)")
    log(f"Selection data     : {threshold_info['selection_data']}")
    log("This is the ONLY threshold in the codebase; app.py reads it from")
    log("models/model_metadata.json.")
    log()

    # ---- 5. Final fit on TRAIN, evaluate ONCE on TEST -------------------
    log("=" * 72)
    log("STEP 5  Final evaluation on the held-out test set")
    log("=" * 72)
    best_pipeline.fit(X_train, y_train)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = compute_metrics(
        y_test.to_numpy(), test_pred, test_proba,
    )
    test_metrics["threshold"] = threshold

    log(f"Threshold: {threshold}")
    log(f"  Accuracy   : {test_metrics['accuracy']}")
    log(f"  Precision  : {test_metrics['precision']}")
    log(f"  Recall     : {test_metrics['recall']}")
    log(f"  F1         : {test_metrics['f1']}")
    log(f"  Specificity: {test_metrics['specificity']}")
    log(f"  ROC-AUC    : {test_metrics['roc_auc']}")
    log(f"  PR-AUC     : {test_metrics['pr_auc']}")
    log(f"  Brier      : {test_metrics['brier']}  (lower is better, "
        f"0 = perfect probabilities)")
    log(f"  ECE        : {test_metrics['ece']}  over "
        f"{cfg.CALIBRATION_BINS} equal-width bins")
    cm = test_metrics["confusion_matrix"]
    log(f"  Confusion  : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} "
        f"TP={cm['tp']}")
    log()

    # ---- 6. Provenance ---------------------------------------------------
    elapsed = round(time.time() - t0, 1)
    result: dict[str, Any] = {
        "artifact_format": "sklearn Pipeline (preprocess + model)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": elapsed,
        "dataset": {
            "path": cfg.DATA_PATH.name,
            "sha256": file_sha256(cfg.DATA_PATH),
            "rows_raw": cleaning_report.rows_raw,
            "rows_final": cleaning_report.rows_final,
            "rows_removed": cleaning_report.total_rows_removed,
        },
        "cleaning_report": cleaning_report.as_dict(),
        "features": {
            "numeric": cfg.NUMERIC_FEATURES,
            "binary": cfg.BINARY_FEATURES,
            "categorical": cfg.CATEGORICAL_FEATURES,
            "model_features": cfg.MODEL_FEATURES,
            "target": cfg.TARGET,
        },
        "split": {
            "test_size": cfg.TEST_SIZE,
            "random_state": SEED,
            "stratified": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "model_selection": {
            "scoring": cfg.SELECTION_SCORING,
            "cv_folds": cfg.CV_FOLDS,
            "cv": "StratifiedKFold(shuffle=True, random_state=42)",
            "selected_model": selected_name,
            "decision": selection,
            "comparison": comparison,
        },
        "calibration": {
            "bins": cfg.CALIBRATION_BINS,
            "test_brier": test_metrics["brier"],
            "test_ece": test_metrics["ece"],
            "mean_predicted": round(float(test_proba.mean()), 4),
            "observed_rate": round(float(y_test.mean()), 4),
        },
        "threshold": threshold_info,
        "test_metrics": test_metrics,
        "provenance": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git_sha": _git_sha(),
        },
        "disclaimer": cfg.DISCLAIMER,
    }

    if save:
        cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, cfg.MODEL_PATH)
        with open(cfg.METADATA_PATH, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        log(f"Saved model     -> {cfg.MODEL_PATH.relative_to(cfg.PROJECT_ROOT)}")
        log(f"Saved metadata  -> {cfg.METADATA_PATH.relative_to(cfg.PROJECT_ROOT)}")

    log()
    log(f"Done in {elapsed}s.")
    return result


def _jsonify(value: Any) -> Any:
    """Make numpy scalars JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    sys.exit(0 if run_training() else 0)
