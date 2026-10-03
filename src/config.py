"""Central configuration for the Student Mental Health pipeline.

Everything that the training pipeline, the Flask app and the notebook must
agree on lives here so there is exactly one source of truth (paths, feature
definitions, category mappings, evaluation constants).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Student Depression Dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CALIBRATION_BINS = 10

# --------------------------------------------------------------------------
# Subgroup (fairness) reporting
# --------------------------------------------------------------------------
# Read-only breakdown of the held-out test set. MIN_N hides groups that are
# too small for a stable estimate rather than reporting a noisy number.
SUBGROUP_COLUMNS = ["Gender", "region", "degree_group"]
AGE_BAND_EDGES = [17, 20, 23, 26, 30, 100]
AGE_BAND_LABELS = ["<=20", "21-23", "24-26", "27-30", "30+"]
SUBGROUP_MIN_N = 30

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
TARGET = "Depression"

# --------------------------------------------------------------------------
# Model features
#
# These are collected by the form in templates/index.html. Every field the UI
# asks for is consumed by the model - there are no decorative inputs.
# --------------------------------------------------------------------------
NUMERIC_FEATURES = ["Age", "CGPA", "Work/Study Hours"]
BINARY_FEATURES = [
    "Suicidal thoughts",
    "Family History of Mental Illness",
    "Gender_Male",
]
CATEGORICAL_FEATURES = [
    "Sleep Duration",
    "Dietary Habits",
    "region",
    "degree_group",
    "Financial_Stress_Category",
    "Study_Satisfaction_Category",
    "Academic_Pressure_Category",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
)

# --------------------------------------------------------------------------
# Data-quality rules
# --------------------------------------------------------------------------
# City strings that appear once or twice in the source file and are obvious
# misspellings / fragments of a real city.
CITY_FIXES = {
    "Khaziabad": "Ghaziabad",
    "Less Delhi": "Delhi",
    "Less than 5 Kalyan": "Kalyan",
    "Nalyan": "Kalyan",
}

REGION_MAP = {
    "South": {
        "Bangalore", "Chennai", "Hyderabad", "Vasai-Virar", "Visakhapatnam",
    },
    "North": {
        "Delhi", "Lucknow", "Srinagar", "Meerut", "Ghaziabad", "Ludhiana",
        "Agra", "Kanpur", "Faridabad",
    },
    "West": {
        "Mumbai", "Thane", "Pune", "Ahmedabad", "Rajkot", "Vadodara",
        "Kalyan", "Nashik", "Jaipur", "Surat",
    },
    "East": {"Kolkata", "Patna", "Varanasi"},
    "Central": {"Indore", "Bhopal", "Nagpur"},
}
CITY_TO_REGION = {
    city: region for region, cities in REGION_MAP.items() for city in cities
}
REGION_LEVELS = list(REGION_MAP) + ["Unknown"]

DEGREE_MAP = {
    "Class 12": "School",
    "BA": "Bachelor", "BSc": "Bachelor", "B.Com": "Bachelor",
    "BBA": "Bachelor", "BCA": "Bachelor", "BE": "Bachelor",
    "B.Tech": "Bachelor", "B.Pharm": "Bachelor", "B.Ed": "Bachelor",
    "BHM": "Bachelor", "B.Arch": "Bachelor", "LLB": "Bachelor",
    "MA": "Master", "MSc": "Master", "M.Com": "Master", "MBA": "Master",
    "M.Ed": "Master", "MCA": "Master", "M.Tech": "Master",
    "M.Pharm": "Master", "ME": "Master", "MHM": "Master",
    "LLM": "Master",
    "PhD": "Doctorate", "MD": "Doctorate",
    "MBBS": "Professional",
}
DEGREE_LEVELS = ["Bachelor", "Doctorate", "Master", "Other", "Professional", "School"]

FINANCIAL_STRESS_MAP = {
    1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High",
}
STUDY_SATISFACTION_MAP = {
    0: "Very Dissatisfied", 1: "Very Dissatisfied", 2: "Dissatisfied",
    3: "Neutral", 4: "Satisfied", 5: "Very Satisfied",
}
ACADEMIC_PRESSURE_MAP = {
    0: "Very Low Pressure", 1: "Very Low Pressure", 2: "Low Pressure",
    3: "Moderate Pressure", 4: "High Pressure", 5: "Very High Pressure",
}

# NOTE: the source survey only offers four sleep bands
# ("Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours")
# plus "Others" - there is no "6-7 hours" level, so the UI must not offer
# one either. See README, "Known dataset gaps".
SLEEP_LEVELS = [
    "5-6 hours", "7-8 hours", "Less than 5 hours",
    "More than 8 hours", "Others",
]
DIET_LEVELS = ["Healthy", "Moderate", "Unhealthy", "Others"]
# NOTE: no "Other" level - the handful of rows with an unusable Financial
# Stress value are removed during cleaning (see data.clean step 10), so the
# fitted encoder only ever sees these five levels.
FINANCIAL_STRESS_LEVELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
STUDY_SATISFACTION_LEVELS = [
    "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied",
]
ACADEMIC_PRESSURE_LEVELS = [
    "Very Low Pressure", "Low Pressure", "Moderate Pressure",
    "High Pressure", "Very High Pressure",
]

# Every legal answer for each categorical form field. Used by the Flask app
# to reject out-of-vocabulary input instead of silently mis-predicting.
ALLOWED_VALUES: dict[str, list[str]] = {
    "Sleep Duration": SLEEP_LEVELS,
    "Dietary Habits": DIET_LEVELS,
    "region": REGION_LEVELS,
    "degree_group": DEGREE_LEVELS,
    "Financial_Stress_Category": FINANCIAL_STRESS_LEVELS,
    "Study_Satisfaction_Category": STUDY_SATISFACTION_LEVELS,
    "Academic_Pressure_Category": ACADEMIC_PRESSURE_LEVELS,
    "Gender": ["Female", "Male"],
    "Suicidal thoughts": ["No", "Yes"],
    "Family History of Mental Illness": ["No", "Yes"],
}

# Numeric input bounds - enforced server side.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "Age": (10, 100),
    "CGPA": (0.0, 10.0),
    "Work/Study Hours": (0.0, 24.0),
}

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
# Model/hyperparameter selection uses threshold-independent ranking quality,
# because the decision threshold is chosen separately afterwards.
SELECTION_SCORING = "roc_auc"

# The single authoritative decision threshold is derived by this objective
# from out-of-fold predictions on the training set. It is NOT a clinical
# cut-off - see README.
THRESHOLD_OBJECTIVE = "f1"

DISCLAIMER = (
    "This is an educational risk-estimation tool built on a public survey "
    "dataset. It does not diagnose depression, anxiety or any other "
    "condition, and it is not a medical device. It is not a substitute for "
    "assessment by a qualified professional."
)
