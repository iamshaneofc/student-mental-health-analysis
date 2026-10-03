# Student Mental Health & Wellbeing — Risk Screening

A Flask + scikit-learn web app that estimates a student's **relative risk of
depression** from a short self-report form, and explains which inputs drove the
score.

> **This is not a diagnosis.** It is an educational risk/screening estimate
> built on a public survey dataset. It does not diagnose depression, anxiety or
> any other condition, it is not a medical device, and it is not a substitute
> for assessment by a qualified professional.

---

## Results

Held-out test set (20% of rows, never touched during model selection,
threshold tuning or calibration decisions):

| Metric | Value |
|---|---|
| ROC-AUC | **0.9216** |
| PR-AUC | 0.9391 |
| Accuracy | 0.8450 |
| Precision | 0.8314 |
| Recall | 0.9221 |
| F1 | 0.8744 |
| Specificity | 0.7362 |
| NPV | 0.8701 |
| Brier (↓, 0 = perfect) | 0.1086 |
| ECE (10 bins, ↓) | 0.0112 |

Confusion matrix @ threshold 0.39 → `TN=1702  FP=610  FN=254  TP=3008`

Model selection ran on the training set only (`n_train = 22,293`,
`n_test = 5,574`), 5-fold stratified CV:

| Candidate | CV ROC-AUC | CV Brier (↓) | Selected config |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.9215 ± 0.0022** | 0.1088 | `C=0.1, penalty='l1', class_weight=None` |
| Gradient Boosting | 0.9206 ± 0.0024 | 0.1096 | `lr=0.05, max_depth=3, n_estimators=300` |
| Random Forest | 0.9159 ± 0.0028 | 0.1158 | `max_depth=12, n_estimators=200` |

Of 24 candidates, **13 were within one standard error of the best AUC**
(floor 0.9205), so ranking quality could not separate them and the decision
came down to calibration. All figures are read from
`models/model_metadata.json`, written by the training run itself — they are
not hand-maintained.

**Calibration:** mean score 0.5807 against an observed rate of 0.5852
(bias −0.0045). The previous configuration (`class_weight='balanced'`)
scored the same 0.9216 AUC but sat **below the observed rate in all 10
bins** (bias −0.0426, ECE 0.0427) — a discrimination-only selection rule
happily traded that away.

**Fairness (held-out test set, same threshold):** gender recall spread
0.0001 / F1 spread 0.0022 — no detectable gap. Age and degree-group spreads
are larger and track differing base rates; see limitation 5 below.

## Methodology

The original notebook had four defects that inflated or destabilised its
numbers. All four are fixed in `src/train.py`:

1. **Preprocessing no longer leaks.** The scaler and encoder are inside an
   sklearn `Pipeline` (impute → scale → one-hot, `drop='first'`,
   `handle_unknown='ignore'`) fitted on training rows only. Previously the
   scaler was fitted on the whole dataset before the split.
2. **One threshold, not two.** There is exactly **one** threshold in the
   codebase — `0.39`, maximising F1 over *out-of-fold* training predictions,
   ties broken toward 0.5. The previous code fitted 0.42 but predicted at 0.45.
   The app reads it from `models/model_metadata.json`.
3. **Model choice is threshold-independent.** Candidates are compared with
   `roc_auc`, so the ranking cannot be an artefact of where the cut-off was
   placed.
4. **Ties on AUC are broken by calibration.** Every candidate within one
   standard error of the best ROC-AUC is still in the running, and the lowest
   out-of-fold Brier score wins. Discrimination first, honest probabilities
   second — both decided on training folds only, with Brier and AUC scored on
   the same fitted folds (no extra CV cost).

Split first (stratified, `random_state=42`), fit on train, evaluate **once** on
test. Cleaning is documented step by step in `CleaningReport` — nothing is
dropped silently.

## Quickstart

Requires Python 3.11+ (developed on 3.12.4).

```bash
git clone https://github.com/iamshaneofc/student-mental-health-analysis.git
cd student-mental-health-analysis

python -m venv .venv
# Windows: .venv\Scripts\activate     POSIX: source .venv/bin/activate

pip install -r requirements.txt
```

The dataset is committed to the repo, and trained artifacts are committed under
`models/`, so you can skip straight to running:

```bash
python app.py                 # http://127.0.0.1:5000
```

To retrain from scratch (≈150 s):

```bash
python -m src.train
```

To run the test suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Endpoints

| Route | Methods | Purpose |
|---|---|---|
| `/` | GET, POST | Assessment form and result |
| `/therapy` | GET | Static list of support/counselling resources |
| `/healthz` | GET | Liveness check — confirms model artifacts loaded |

Server-side validation rejects out-of-range numbers and unrecognised choices
with **HTTP 400** and an inline error, never a 500. Form values are never
written to the log; only the resulting risk band is.

Debug mode is **off** unless you explicitly set `FLASK_DEBUG=1`.

## Project structure

```
.
├── app.py                       Flask app: validate → estimate → risk band
├── main.ipynb                   Executable report (imports src/, no copy-paste)
├── src/
│   ├── config.py                Paths, feature lists, category maps, bounds
│   ├── data.py                  Cleaning + CleaningReport + sha256
│   └── train.py                 Pipeline, selection, threshold, metrics, subgroup check
├── models/
│   ├── model.joblib             sklearn Pipeline (preprocess + model)
│   └── model_metadata.json      Metrics, threshold, decision record, provenance
├── templates/                   index.html (form), therapy.html (resources)
├── static/                      CSS and images
├── tests/test_smoke.py          30 smoke tests
├── requirements.txt             Runtime (pinned)
├── requirements-dev.txt         Runtime + jupyter/pytest/plotting
└── Student Depression Dataset.csv
```

`main.ipynb` is a **report over the same code the app runs** — it imports
`src.data` and calls `src.train.run_training(save=False)` rather than
re-implementing anything, so the analysis cannot drift away from production.
Verified to execute top-to-bottom in a fresh kernel.

## Data

["Student Depression Dataset" (Kaggle)](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
— 27,901 rows × 18 columns, self-reported, India-centric.

Cleaning removes **34 rows (0.122%)** in 11 recorded steps:

| Step | Change | Rows |
|---|---|---|
| 1 | Strip whitespace, fix 4 city misspellings | 0 |
| 2 | Derive `region` from `City` (22 junk cities kept as `Unknown`) | 0 |
| 3 | Keep `Profession == 'Student'` | −31 |
| 4 | Encode Yes/No + gender as 0/1 | 0 |
| 5 | Normalise sleep strings (18 `'Others'` **kept**) | 0 |
| 6 | Normalise diet strings (12 `'Others'` **kept**) | 0 |
| 7 | Group 28 raw degrees into 6 `degree_group`s | 0 |
| 8 | Coerce ordinal scales to integers (3 bad `Financial Stress`) | 0 |
| 9 | Map 0–5 scales to named levels | 0 |
| 10 | Drop rows missing `Financial Stress` | −3 |
| 11 | Drop `Work Pressure`, `Job Satisfaction` (>99.97% constant) | 0 |

13 model features: `Age`, `CGPA`, `Work/Study Hours`, `Suicidal thoughts`,
`Family History of Mental Illness`, `Gender_Male`, `Sleep Duration`,
`Dietary Habits`, `region`, `degree_group`, `Financial_Stress_Category`,
`Study_Satisfaction_Category`, `Academic_Pressure_Category`.

## Known limitations

These are properties of the project, not footnotes:

1. **No clinical ground truth.** The target is a self-reported survey label,
   never validated against a clinician's assessment.
2. **Construct overlap.** `Suicidal thoughts` is the strongest feature *and* is
   closely associated with the label being predicted, so performance is
   optimistic for any pre-symptom screening reading.
3. **Calibrated here, not calibrated in general.** Test ECE 0.0112 and mean
   score 0.5807 vs observed 0.5852 mean the scores track observed rates
   *within this dataset*. That says nothing about a different population, and
   the score is still **not** a probability of illness.
4. **Single dataset, single geography, no external validation.**
5. **Fairness is spot-checked, not audited.** On the held-out test set at
   threshold 0.39, gender performance is effectively identical (recall spread
   0.0001, F1 spread 0.0022), consistent with `Gender_Male` being shrunk to
   exactly 0 by L1. Larger spreads appear across **age band** (recall
   0.8551–0.9613, specificity 0.6066–0.8409) and **degree group** (recall
   0.8440–0.9517) — but those groups also have very different base rates
   (prevalence 0.4192 for 30+ vs 0.7179 for ≤20), so a single global
   threshold necessarily trades specificity for recall between them. No
   formal fairness criterion (equalised odds, within-group calibration) is
   enforced, and the slices are not corrected for multiple comparisons.
6. **The threshold is tuned on training folds**, so the operating point is
   mildly optimistic by construction.
7. **Single threshold, single objective.** `0.39` maximises F1; a use case
   that cares more about missing an at-risk student than about false alarms
   would want a different operating point, chosen the same way.
8. **Selection spends its tie-break on calibration.** Candidates are ranked by
   AUC first, and only candidates within one standard error of the best
   compete on Brier. A future run with a genuinely better-ranked model more
   than one SE ahead still wins outright.

## Roadmap

Phase 1 (this state) is a clean, reproducible baseline. Later phases may add a
dedicated wellbeing questionnaire, richer interpretation of results, and
report export. No such feature exists today.

## References

- Kaggle. *Student Depression Dataset*.
  <https://www.kaggle.com/datasets/hopesb/student-depression-dataset>
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  *JMLR*, 12, 2825–2830.
- Flask documentation. <https://flask.palletsprojects.com/>
