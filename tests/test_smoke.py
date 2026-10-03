"""Smoke tests for the Student Mental Health app and model artifact.

Run from the project root::

    python -m pytest -q
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pytest

import app as app_module
from src import config as cfg
from src.data import load_and_clean, model_feature_frame

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def valid_payload() -> dict[str, str]:
    return {
        "Age": "21",
        "CGPA": "8.2",
        "Work_Study_Hours": "8",
        "Suicidal_thoughts": "No",
        "Family_History": "No",
        "Gender": "Female",
        "Sleep_Duration": "7-8 hours",
        "Dietary_Habits": "Healthy",
        "region": "South",
        "degree_group": "Bachelor",
        "Financial_Stress": "Low",
        "Study_Satisfaction": "Satisfied",
        "Academic_Pressure": "Low Pressure",
    }


def high_risk_payload(valid_payload: dict[str, str]) -> dict[str, str]:
    """Same form, answers associated with higher model-estimated risk."""
    p = dict(valid_payload)
    p.update({
        "Suicidal_thoughts": "Yes",
        "Family_History": "Yes",
        "Sleep_Duration": "Less than 5 hours",
        "Dietary_Habits": "Unhealthy",
        "Financial_Stress": "Very High",
        "Study_Satisfaction": "Very Dissatisfied",
        "Academic_Pressure": "Very High Pressure",
    })
    return p


# --------------------------------------------------------------------------
# Artifacts / provenance
# --------------------------------------------------------------------------


def test_artifacts_exist():
    assert cfg.MODEL_PATH.exists(), "run `python -m src.train` first"
    assert cfg.METADATA_PATH.exists()


def test_metadata_is_self_consistent():
    meta = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["features"]["model_features"] == cfg.MODEL_FEATURES
    assert 0.0 < meta["threshold"]["threshold"] < 1.0
    assert meta["split"]["stratified"] is True
    assert meta["dataset"]["sha256"]
    # Selection must have used training data only.
    assert meta["threshold"]["selection_data"].startswith("out-of-fold")


def test_single_authoritative_threshold():
    """There must be exactly one threshold, read from the metadata."""
    meta = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    assert app_module.THRESHOLD == meta["threshold"]["threshold"]
    # No other hard-coded threshold may appear in the app source.
    source = (cfg.PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    for banned in ("0.45", "0.42", "> 0.5", "== 0.5"):
        assert banned not in source, f"hard-coded threshold {banned!r} in app.py"


def test_pipeline_artifact_matches_config():
    pipeline = joblib.load(cfg.MODEL_PATH)
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert len(names) == pipeline.named_steps["model"].n_features_in_


def test_report_metrics_present():
    meta = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    for key in ("accuracy", "precision", "recall", "f1",
                "specificity", "roc_auc", "pr_auc"):
        assert key in meta["test_metrics"], key
    assert set(meta["test_metrics"]["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}


# --------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------


def test_cleaning_drops_almost_nothing():
    df, report = load_and_clean()
    assert report.rows_raw == 27901
    assert report.rows_final == len(df)
    assert report.removal_rate < 0.01  # under 1%
    assert report.total_rows_removed == 34


def test_no_missing_values_after_cleaning():
    df, _ = load_and_clean()
    assert int(df[cfg.MODEL_FEATURES].isna().sum().sum()) == 0


def test_feature_frame_column_order_is_stable():
    values = {
        "Age": 21.0, "CGPA": 8.2, "Work/Study Hours": 8.0,
        "Suicidal thoughts": "No",
        "Family History of Mental Illness": "No",
        "Gender": "Female",
        "Sleep Duration": "7-8 hours",
        "Dietary Habits": "Healthy",
        "region": "South",
        "degree_group": "Bachelor",
        "Financial_Stress_Category": "Low",
        "Study_Satisfaction_Category": "Satisfied",
        "Academic_Pressure_Category": "Low Pressure",
    }
    frame = model_feature_frame(values)
    assert list(frame.columns) == cfg.MODEL_FEATURES
    assert int(frame["Gender_Male"].iloc[0]) == 0
    assert int(frame["Suicidal thoughts"].iloc[0]) == 0


def test_every_form_level_exists_in_training_data():
    """The UI must never offer an answer the model has never seen."""
    df, _ = load_and_clean()
    for feature in cfg.CATEGORICAL_FEATURES:
        seen = set(df[feature].unique())
        offered = set(cfg.ALLOWED_VALUES[feature])
        missing = offered - seen
        assert not missing, f"{feature}: form offers untrained levels {missing}"


# --------------------------------------------------------------------------
# HTTP behaviour
# --------------------------------------------------------------------------


def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_get_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"<form" in res.data


def test_get_therapy(client):
    res = client.get("/therapy")
    assert res.status_code == 200


def test_therapy_renders_disclaimer(client):
    """Regression test: the route passes `disclaimer` but the template never
    rendered it, so the resources page shipped with no non-diagnostic note."""
    res = client.get("/therapy")
    assert b"does not diagnose" in res.data


def test_no_diagnostic_framing_in_templates():
    """No template may claim to diagnose, cure, or report a real condition."""
    banned = [
        "You are healthy",
        "Depression Risk Detected",
        "Back to Prediction",
        "chance of depression",
        "MindSpace",
    ]
    for name in ("index.html", "therapy.html"):
        source = (cfg.TEMPLATES_DIR / name).read_text(encoding="utf-8")
        for phrase in banned:
            assert phrase not in source, f"{name} still contains {phrase!r}"


def test_valid_submission_lower_risk(client, valid_payload):
    res = client.post("/", data=valid_payload)
    assert res.status_code == 200
    assert b"Lower model-estimated risk" in res.data


def test_valid_submission_elevated_risk(client, valid_payload):
    res = client.post("/", data=high_risk_payload(valid_payload))
    assert res.status_code == 200
    assert b"Elevated model-estimated risk" in res.data


def test_disclaimer_is_rendered(client, valid_payload):
    res = client.post("/", data=valid_payload)
    assert b"does not diagnose" in res.data


def test_non_numeric_age_fails_gracefully(client, valid_payload):
    """Regression test: this used to raise an unhandled 500."""
    payload = dict(valid_payload, Age="abc")
    res = client.post("/", data=payload)
    assert res.status_code == 400
    assert b"must be a number" in res.data


def test_missing_age_fails_gracefully(client, valid_payload):
    payload = dict(valid_payload, Age="")
    res = client.post("/", data=payload)
    assert res.status_code == 400


def test_out_of_range_cgpa_rejected(client, valid_payload):
    payload = dict(valid_payload, CGPA="99")
    res = client.post("/", data=payload)
    assert res.status_code == 400
    assert b"between 0 and 10" in res.data


def test_unknown_choice_value_rejected(client, valid_payload):
    payload = dict(valid_payload, region="Atlantis")
    res = client.post("/", data=payload)
    assert res.status_code == 400


def test_bad_input_does_not_echo_payload_into_prediction(client, valid_payload):
    payload = dict(valid_payload, Age="not-a-number")
    res = client.post("/", data=payload)
    assert res.status_code == 400
    # The entered value is echoed back so the user can correct it...
    assert b"not-a-number" in res.data
    # ...but no prediction block is produced.
    assert b"model risk score" not in res.data.lower()


# --------------------------------------------------------------------------
# Privacy: sensitive responses must never be logged
# --------------------------------------------------------------------------


def test_form_values_are_never_logged(client, valid_payload, caplog):
    marker_age, marker_cgpa, marker_hours = "33.0", "7.77", "11.11"
    payload = dict(
        valid_payload, Age=marker_age, CGPA=marker_cgpa,
        Work_Study_Hours=marker_hours,
    )
    with caplog.at_level(logging.DEBUG):
        res = client.post("/", data=payload)
    assert res.status_code == 200

    logged = "\n".join(r.getMessage() for r in caplog.records)
    for marker in (marker_age, marker_cgpa, marker_hours):
        assert marker not in logged, f"sensitive form value leaked into logs: {marker}"
    assert not any(r.levelno < logging.INFO for r in caplog.records), (
        "app must not log at DEBUG"
    )


def test_no_debug_flag_in_source():
    source = (cfg.PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "debug=True" not in source
    assert "app.logger.debug" not in source
    assert "logging.DEBUG" not in source


# --------------------------------------------------------------------------
# Fairness: read-only subgroup check on the held-out test set
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def subgroup_report():
    from src.train import subgroup_metrics

    pipeline = joblib.load(cfg.MODEL_PATH)
    threshold = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))[
        "threshold"
    ]["threshold"]
    return subgroup_metrics(pipeline, threshold=threshold)


def test_gender_parity_guard(subgroup_report):
    """Regression guard: a gender gap opening up should fail loudly.

    Measured 2026-10-03 at threshold 0.39: recall spread 0.0001,
    F1 spread 0.0022.
    """
    frame = subgroup_report["Gender"]
    assert set(frame.index) == {"Female", "Male"}
    assert frame["recall"].max() - frame["recall"].min() < 0.02
    assert frame["f1"].max() - frame["f1"].min() < 0.01


def test_subgroup_slices_cover_the_test_set(subgroup_report):
    expected = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))["split"][
        "n_test"
    ]
    for name, frame in subgroup_report.items():
        assert frame["n"].sum() == expected, f"{name} does not cover the test set"


def test_subgroup_small_groups_are_not_scored(subgroup_report):
    """Groups too small for a stable estimate get NaN, never a guess."""
    for frame in subgroup_report.values():
        unscored = frame[frame["recall"].isna()]
        assert (unscored["n"] < cfg.SUBGROUP_MIN_N).all()


def test_subgroup_metrics_does_not_refit():
    """Read-only by construction: the artifact must be untouched."""
    before = cfg.MODEL_PATH.read_bytes()
    pipeline = joblib.load(cfg.MODEL_PATH)
    from src.train import subgroup_metrics

    subgroup_metrics(pipeline, threshold=0.5)
    assert cfg.MODEL_PATH.read_bytes() == before


# --------------------------------------------------------------------------
# Calibration: the score shown to a user must track observed rates
# --------------------------------------------------------------------------


def test_calibration_is_held_out_and_honest():
    """Regression guard against the bias class_weight='balanced' introduced.

    Measured 2026-10-03: Brier 0.1086, ECE 0.0112, mean score 0.5807 against
    an observed rate of 0.5852. The old configuration scored 0.1113 / 0.0427
    with a -0.0426 bias, and was wrong in the same direction in all 10 bins.
    """
    from src.train import held_out_split

    calibration = json.loads(
        cfg.METADATA_PATH.read_text(encoding="utf-8")
    )["calibration"]
    assert calibration["test_brier"] < 0.12
    assert calibration["test_ece"] < 0.03
    assert abs(calibration["mean_predicted"] - calibration["observed_rate"]) < 0.02

    pipeline = joblib.load(cfg.MODEL_PATH)
    X_test, y_test = held_out_split()
    proba = pipeline.predict_proba(X_test)[:, 1]

    # A systematic one-directional bias would show up as every bin's
    # observed rate sitting above the mean score.
    edges = np.linspace(0, 1, calibration["bins"] + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, calibration["bins"] - 1)
    gaps = [
        float(y_test.values[idx == b].mean() - proba[idx == b].mean())
        for b in range(calibration["bins"])
        if (idx == b).any()
    ]
    assert not all(g > 0 for g in gaps), "every bin under-predicts again"


def test_selection_record_matches_artifact():
    """The decision recorded in metadata must describe what was saved."""
    metadata = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    decision = metadata["model_selection"]["decision"]
    assert decision["selected_model"] == metadata["model_selection"][
        "selected_model"
    ]
    assert decision["n_eligible"] <= decision["n_candidates"]
    assert decision["eligibility_floor"] <= decision["best_cv_roc_auc"]
    assert metadata["test_metrics"]["brier"] == metadata["calibration"][
        "test_brier"
    ]
