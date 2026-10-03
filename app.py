"""Flask application for the Student Mental Health risk-screening demo.

Loads a single versioned artifact (``models/model.joblib``) plus its
metadata, validates input, and returns a *model-based risk estimate*.

IMPORTANT: this tool estimates risk from a model trained on a public survey
dataset. It does not diagnose depression or any other condition.

Logging policy: form responses contain sensitive mental-health disclosures
and are therefore NEVER logged. Only lifecycle events and non-identifying
outcomes (validation failure counts, status codes) are recorded.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request

from src import config as cfg
from src.data import model_feature_frame

# --------------------------------------------------------------------------
# Logging - information level by default, never DEBUG in committed code.
# --------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smh.app")

app = Flask(__name__)

# --------------------------------------------------------------------------
# Model loading (path anchored to this file, not to the CWD)
# --------------------------------------------------------------------------


def _load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not cfg.MODEL_PATH.exists() or not cfg.METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Expected {cfg.MODEL_PATH} and "
            f"{cfg.METADATA_PATH}. Run `python -m src.train` first."
        )
    pipeline = joblib.load(cfg.MODEL_PATH)
    with open(cfg.METADATA_PATH, encoding="utf-8") as fh:
        metadata = json.load(fh)
    return pipeline, metadata


MODEL, METADATA = _load_artifacts()
THRESHOLD: float = float(METADATA["threshold"]["threshold"])
DISCLAIMER: str = METADATA.get("disclaimer", cfg.DISCLAIMER)
logger.info(
    "Loaded model artifact (model=%s, threshold=%.2f, dataset=%s)",
    METADATA["model_selection"]["selected_model"],
    THRESHOLD,
    METADATA["dataset"]["sha256"][:12],
)

# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

# form field name -> (model feature name, kind, spec)
NUMERIC_FIELDS: dict[str, tuple[str, tuple[float, float]]] = {
    "Age": ("Age", cfg.NUMERIC_BOUNDS["Age"]),
    "CGPA": ("CGPA", cfg.NUMERIC_BOUNDS["CGPA"]),
    "Work_Study_Hours": ("Work/Study Hours", cfg.NUMERIC_BOUNDS["Work/Study Hours"]),
}

CHOICE_FIELDS: dict[str, str] = {
    "Suicidal_thoughts": "Suicidal thoughts",
    "Family_History": "Family History of Mental Illness",
    "Gender": "Gender",
    "Sleep_Duration": "Sleep Duration",
    "Dietary_Habits": "Dietary Habits",
    "region": "region",
    "degree_group": "degree_group",
    "Financial_Stress": "Financial_Stress_Category",
    "Study_Satisfaction": "Study_Satisfaction_Category",
    "Academic_Pressure": "Academic_Pressure_Category",
}


def _coerce_number(raw: Any, low: float, high: float) -> tuple[float | None, str | None]:
    """Return (value, error). Never raises."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None, "is required."
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None, "must be a number."
    if value != value:  # NaN
        return None, "must be a number."
    if not (low <= value <= high):
        return None, f"must be between {low:g} and {high:g}."
    return value, None


def validate_form(form) -> tuple[dict[str, Any] | None, list[str], dict[str, str]]:
    """Validate raw form input.

    Returns ``(model_form, errors, echo)`` where ``model_form`` is the
    normalised mapping consumed by :func:`model_feature_frame`, ``errors``
    is a list of human-readable messages, and ``echo`` holds the submitted
    values so they can be repopulated after a failure.
    """
    errors: list[str] = []
    echo: dict[str, str] = {}
    values: dict[str, Any] = {}

    for field, (feature, (low, high)) in NUMERIC_FIELDS.items():
        raw = form.get(field, "")
        echo[field] = str(raw)
        value, err = _coerce_number(raw, low, high)
        if err:
            label = field.replace("_", " ").replace("Work Study", "Work/Study")
            errors.append(f"{label} {err}")
        else:
            values[feature] = value

    for field, feature in CHOICE_FIELDS.items():
        raw = form.get(field, "")
        echo[field] = str(raw)
        allowed = cfg.ALLOWED_VALUES[feature]
        if raw not in allowed:
            errors.append(
                f"{field.replace('_', ' ')} must be one of: "
                f"{', '.join(allowed)}."
            )
        else:
            values[feature] = raw

    if errors:
        return None, errors, echo
    return values, [], echo


def estimate(values: dict[str, Any]) -> tuple[str, float]:
    """Run the artifact. Returns ``(label, probability)``."""
    frame = model_feature_frame(values)
    probability = float(MODEL.predict_proba(frame)[0][1])
    label = "Yes" if probability >= THRESHOLD else "No"
    return label, probability


def risk_band(probability: float) -> str:
    return "elevated" if probability >= THRESHOLD else "lower"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/healthz")
def healthz():
    """Liveness probe - no user data involved."""
    return {
        "status": "ok",
        "model": METADATA["model_selection"]["selected_model"],
        "threshold": THRESHOLD,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    context: dict[str, Any] = {
        "prediction": None,
        "probability": None,
        "risk_band": None,
        "threshold": THRESHOLD,
        "disclaimer": DISCLAIMER,
        "errors": [],
        "form_data": {},
    }

    if request.method == "POST":
        values, errors, echo = validate_form(request.form)
        context["form_data"] = echo
        if errors:
            # Malformed input must never reach the estimator.
            context["errors"] = errors
            logger.warning(
                "Rejected submission with %d validation error(s)", len(errors)
            )
            return render_template("index.html", **context), 400

        try:
            label, probability = estimate(values)
        except Exception:
            # Do not echo the payload - it contains sensitive disclosures.
            logger.exception("Prediction failed")
            context["errors"] = [
                "The assessment could not be computed. Please try again."
            ]
            return render_template("index.html", **context), 500

        context.update(
            prediction=label,
            probability=round(probability * 100, 2),
            risk_band=risk_band(probability),
        )
        logger.info("Assessment completed (band=%s)", context["risk_band"])

    return render_template("index.html", **context)


@app.get("/therapy")
def therapy():
    return render_template("therapy.html", disclaimer=DISCLAIMER)


if __name__ == "__main__":
    # Never enable the Werkzeug debugger from committed code.
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=debug)
