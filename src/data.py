"""Dataset loading and explicit, auditable data cleaning.

Every transformation records what it did and how many rows it touched, so
the cleaning step can be inspected rather than guessed at. Nothing is
dropped without appearing in the returned cleaning report.

Row filters are applied *before* the train/test split. They are driven only
by feature values (never by the target), so they do not leak label
information into the test set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass
class CleaningStep:
    """One auditable transformation applied to the dataset."""

    step: str
    description: str
    rows_before: int
    rows_after: int
    rows_removed: int
    columns_added: list[str] = field(default_factory=list)
    columns_changed: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
            "columns_added": self.columns_added,
            "columns_changed": self.columns_changed,
            "notes": self.notes,
        }


@dataclass
class CleaningReport:
    steps: list[CleaningStep] = field(default_factory=list)
    rows_raw: int = 0
    rows_final: int = 0

    def add(self, step: CleaningStep) -> None:
        self.steps.append(step)

    @property
    def total_rows_removed(self) -> int:
        return self.rows_raw - self.rows_final

    @property
    def removal_rate(self) -> float:
        return self.total_rows_removed / self.rows_raw if self.rows_raw else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_raw": self.rows_raw,
            "rows_final": self.rows_final,
            "total_rows_removed": self.total_rows_removed,
            "removal_rate": round(self.removal_rate, 6),
            "steps": [s.as_dict() for s in self.steps],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.as_dict() for s in self.steps])

    def summary(self) -> str:
        lines = [
            f"Rows: {self.rows_raw:,} raw -> {self.rows_final:,} final "
            f"({self.total_rows_removed:,} removed, "
            f"{self.removal_rate * 100:.3f}%)",
            "",
        ]
        for s in self.steps:
            delta = s.rows_after - s.rows_before
            delta_txt = f"{delta:+d}" if delta else "  0"
            lines.append(f"  {s.step:<3} {s.description}")
            lines.append(f"       rows {s.rows_before:,} -> {s.rows_after:,} "
                         f"({delta_txt})")
            if s.notes:
                lines.append(f"       note: {s.notes}")
        return "\n".join(lines)


def file_sha256(path) -> str:
    """SHA-256 of a file, used to pin the dataset the model was trained on."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path=None) -> pd.DataFrame:
    """Read the source CSV without modifying it."""
    path = path or cfg.DATA_PATH
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the documented cleaning pipeline.

    Returns the cleaned frame plus a report of every transformation.
    """
    report = CleaningReport(rows_raw=len(df))
    df = df.copy()

    # -- 1. Normalise city spelling ---------------------------------------
    before = len(df)
    df["City"] = (
        df["City"].astype(str).str.strip(" \t\r\n'\"").replace(cfg.CITY_FIXES)
    )
    report.add(CleaningStep(
        step=1,
        description="Strip whitespace/quotes from City and fix 4 misspellings",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_changed=["City"],
        notes=("Fixed: Khaziabad->Ghaziabad, Less Delhi->Delhi, "
               "Nalyan->Kalyan, Less than 5 Kalyan->Kalyan"),
    ))

    # -- 2. City -> region ------------------------------------------------
    before = len(df)
    df["region"] = df["City"].map(cfg.CITY_TO_REGION).fillna("Unknown")
    unmapped = int((df["region"] == "Unknown").sum())
    report.add(CleaningStep(
        step=2,
        description="Derive region from City (5 Indian regions)",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_added=["region"],
        notes=(f"{unmapped} rows have a city not in the mapping "
               f"(free-text junk such as names or 'City'); retained as the "
               f"explicit level 'Unknown' rather than discarded."),
    ))

    # -- 3. Students only -------------------------------------------------
    before = len(df)
    non_student = df["Profession"] != "Student"
    dropped_prof = df.loc[non_student, "Profession"].value_counts().to_dict()
    df = df.loc[~non_student].copy()
    report.add(CleaningStep(
        step=3,
        description="Keep rows where Profession == 'Student'",
        rows_before=before,
        rows_after=len(df),
        rows_removed=before - len(df),
        notes=(f"The product targets students, so non-student rows are out "
               f"of scope. Removed professions: {dropped_prof}"),
    ))

    # -- 4. Binary encodings ---------------------------------------------
    before = len(df)
    df["Suicidal thoughts"] = (
        df["Have you ever had suicidal thoughts ?"].astype(str).str.strip()
        .eq("Yes").astype(int)
    )
    df["Family History of Mental Illness"] = (
        df["Family History of Mental Illness"].astype(str).str.strip()
        .eq("Yes").astype(int)
    )
    df["Gender_Male"] = df["Gender"].astype(str).str.strip().eq("Male").astype(int)
    report.add(CleaningStep(
        step=4,
        description="Encode Yes/No disclosures and gender as 0/1 integers",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_added=[
            "Suicidal thoughts", "Family History of Mental Illness", "Gender_Male",
        ],
    ))

    # -- 5. Sleep duration -------------------------------------------------
    before = len(df)
    df["Sleep Duration"] = df["Sleep Duration"].astype(str).str.strip(" \t\r\n'\"")
    n_other = int((df["Sleep Duration"] == "Others").sum())
    report.add(CleaningStep(
        step=5,
        description="Normalise Sleep Duration strings",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_changed=["Sleep Duration"],
        notes=(f"{n_other} rows answer 'Others'; they are RETAINED as an "
               f"explicit level instead of being dropped (the original "
               f"notebook silently discarded them)."),
    ))

    # -- 6. Dietary habits -------------------------------------------------
    before = len(df)
    df["Dietary Habits"] = df["Dietary Habits"].astype(str).str.strip(" \t\r\n'\"")
    report.add(CleaningStep(
        step=6,
        description="Normalise Dietary Habits strings",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_changed=["Dietary Habits"],
        notes=f"{int((df['Dietary Habits'] == 'Others').sum())} rows answer "
              f"'Others'; retained as an explicit level.",
    ))

    # -- 7. Degree -> degree_group ----------------------------------------
    before = len(df)
    df["degree_group"] = df["Degree"].map(cfg.DEGREE_MAP).fillna("Other")
    report.add(CleaningStep(
        step=7,
        description="Group the 28 raw Degree values into 6 degree groups",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_added=["degree_group"],
        notes=f"Unmapped degrees -> 'Other': "
              f"{int((df['degree_group'] == 'Other').sum())} rows.",
    ))

    # -- 8. Coerce the three ordinal scales to integers --------------------
    before = len(df)
    for col in ("Financial Stress", "Study Satisfaction", "Academic Pressure"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    n_fin_na = int(df["Financial Stress"].isna().sum())
    n_sat_na = int(df["Study Satisfaction"].isna().sum())
    n_pac_na = int(df["Academic Pressure"].isna().sum())
    report.add(CleaningStep(
        step=8,
        description="Coerce Financial Stress / Study Satisfaction / "
                    "Academic Pressure to integers",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_changed=[
            "Financial Stress", "Study Satisfaction", "Academic Pressure",
        ],
        notes=(f"Non-numeric or absent values -> NA: "
               f"Financial Stress={n_fin_na}, "
               f"Study Satisfaction={n_sat_na}, "
               f"Academic Pressure={n_pac_na}."),
    ))

    # -- 9. Ordinal scale -> readable category ----------------------------
    before = len(df)
    df["Financial_Stress_Category"] = (
        df["Financial Stress"].map(cfg.FINANCIAL_STRESS_MAP).fillna("Other")
    )
    df["Study_Satisfaction_Category"] = df["Study Satisfaction"].map(
        cfg.STUDY_SATISFACTION_MAP
    )
    df["Academic_Pressure_Category"] = df["Academic Pressure"].map(
        cfg.ACADEMIC_PRESSURE_MAP
    )
    report.add(CleaningStep(
        step=9,
        description="Map the three 0-5 / 1-5 ordinal scales to named levels",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        columns_added=[
            "Financial_Stress_Category",
            "Study_Satisfaction_Category",
            "Academic_Pressure_Category",
        ],
    ))

    # -- 10. Missing Financial Stress --------------------------------------
    # The category mapping above turns missing values into the level "Other",
    # which for 3 rows is indistinguishable from a real answer, so the rows
    # are removed explicitly rather than relabelled.
    before = len(df)
    missing_stress = df["Financial Stress"].isna()
    df = df.loc[~missing_stress].copy()
    report.add(CleaningStep(
        step=10,
        description="Remove rows with a missing Financial Stress value",
        rows_before=before,
        rows_after=len(df),
        rows_removed=before - len(df),
        notes=("Listwise deletion on a primary feature (0.01% of the file). "
               "The original notebook did the same, but only after silently "
               "relabelling the value."),
    ))

    # -- 11. Drop columns that carry no usable signal ----------------------
    before = len(df)
    before_cols = list(df.columns)
    dead_cols = ["Work Pressure", "Job Satisfaction"]
    df = df.drop(columns=[c for c in dead_cols if c in df.columns])
    report.add(CleaningStep(
        step=11,
        description="Drop constant / near-constant columns",
        rows_before=before,
        rows_after=len(df),
        rows_removed=0,
        notes=(f"Dropped {dead_cols}: they are 0 for >99.97% of rows "
               f"(survey did not ask students). Columns "
               f"{len(before_cols)} -> {len(df.columns)}."),
    ))

    report.rows_final = len(df)
    df = df.reset_index(drop=True)
    return df, report


def load_and_clean(path=None) -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience wrapper: read the CSV and clean it."""
    return clean(load_raw(path))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the model feature columns, in canonical order."""
    missing = [c for c in cfg.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model feature columns: {missing}")
    return df[cfg.MODEL_FEATURES].copy()


def target_series(df: pd.DataFrame) -> pd.Series:
    """Return the target column."""
    if cfg.TARGET not in df.columns:
        raise KeyError(f"Missing target column: {cfg.TARGET}")
    return df[cfg.TARGET].astype(int)


def model_feature_frame(form: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row frame containing exactly ``MODEL_FEATURES``.

    The form already submits *post-mapping* answers (region instead of City,
    degree_group instead of Degree, named Likert levels instead of integers),
    so the app maps straight onto the model features. This is the only place
    that translation happens, and it is validated by ``tests/test_smoke.py``.
    """
    row = {
        "Age": form["Age"],
        "CGPA": form["CGPA"],
        "Work/Study Hours": form["Work/Study Hours"],
        "Suicidal thoughts": 1 if form["Suicidal thoughts"] == "Yes" else 0,
        "Family History of Mental Illness": (
            1 if form["Family History of Mental Illness"] == "Yes" else 0
        ),
        "Gender_Male": 1 if form["Gender"] == "Male" else 0,
        "Sleep Duration": form["Sleep Duration"],
        "Dietary Habits": form["Dietary Habits"],
        "region": form["region"],
        "degree_group": form["degree_group"],
        "Financial_Stress_Category": form["Financial_Stress_Category"],
        "Study_Satisfaction_Category": form["Study_Satisfaction_Category"],
        "Academic_Pressure_Category": form["Academic_Pressure_Category"],
    }
    return pd.DataFrame([row], columns=cfg.MODEL_FEATURES)
