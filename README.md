# Diabetes Risk Predictor

Predicts whether a patient is likely to have diabetes from 8 clinical
measurements, using the classic Pima Indians Diabetes dataset (768 patients:
268 diabetic, 500 not diabetic).

**Educational demo only — this is not medical advice and not a medical
diagnosis.** Predictions can be wrong; anyone concerned about diabetes should
see a healthcare professional.

## What is in this repo

| File | What it is |
| --- | --- |
| `app.py` | Streamlit frontend (repo root, for Streamlit Cloud) |
| `notebooks/diabetes_model.ipynb` | Backend notebook — cleaning, both modelling iterations, tuning, final report (fully executed, outputs saved) |
| `data/diabetes.csv` | Raw dataset as uploaded |
| `data/diabetes_cleaned.csv` | Cleaned dataset (see below) |
| `requirements.txt` | Python dependencies |

No binary model file is included on purpose: `app.py` retrains the chosen
pipeline on the cleaned data at startup (the dataset is tiny, so this takes
only a few seconds) and therefore stays within text-only, GitHub-friendly files.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then enter the 8 measurements (defaults are the dataset medians) and press
**Predict**. The app shows the prediction, the predicted probability, a
plain-language interpretation, the model's test metrics, and the top feature
importances.

## Data cleaning

A living patient cannot have a Glucose, BloodPressure, SkinThickness, Insulin
or BMI of 0, so zeros in those columns were treated as missing data:

| Column | Impossible zeros | Share of rows |
| --- | --- | --- |
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

- 652 impossible-zero cells in total were converted to missing and
  median-imputed. Modelling pipelines impute from **training data only**
  (leakage-safe); the saved `diabetes_cleaned.csv` is median-imputed on the
  full data and is labelled as such in the notebook.
- **No rows were dropped** — dropping every affected row would have discarded
  most of the dataset.
- **0 duplicate rows** were found.

## Modelling

Stratified 80/20 train/test split (`random_state=42` throughout); test set =
154 patients, 54 diabetic. Models were selected primarily by test **F1**,
with **Recall** reported explicitly because missing a diabetic patient is the
costly error for a screening tool.

### Iteration 1 — baselines (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.7662 | 0.6957 | 0.5926 | 0.6400 | 0.8210 |
| KNN | 0.7532 | 0.6600 | 0.6111 | 0.6346 | 0.7899 |
| Gradient Boosting | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8309 |
| SVM | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 |
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |

### Iteration 2 — tuned top 2 (5-fold GridSearchCV on the training set, scoring F1)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest (tuned) | 0.7468 | 0.6056 | 0.7963 | 0.6880 | 0.8185 |
| KNN (tuned) | 0.7338 | 0.6327 | 0.5741 | 0.6019 | 0.7944 |

**Iteration 2 beat Iteration 1:** best F1 rose **+0.0480** (0.6400 → 0.6880) and
Recall rose **+0.2037** (0.5926 → 0.7963). Tuning helped Random Forest a lot
(mostly via `class_weight='balanced'`) but did not help KNN — tuned KNN was
worse than baseline KNN (F1 0.6019 vs 0.6346).

## Chosen best model

**Tuned Random Forest** — `n_estimators=200, max_depth=5, min_samples_split=5,
class_weight='balanced'`, 5-fold CV F1 0.6998 on the training folds.

| Metric | Test value |
| --- | --- |
| Accuracy | 0.7468 |
| Precision | 0.6056 |
| Recall | 0.7963 |
| F1 | 0.6880 |
| ROC-AUC | 0.8185 |

Confusion matrix (test set): True Negatives 72, False Positives 28,
False Negatives 11, True Positives 43. So 11 of 54 actually diabetic patients
were missed, and 28 healthy patients were flagged — the trade-off that comes
with the higher recall.

Top feature importances: Glucose 0.3308, BMI 0.1858, Age 0.1239,
Insulin 0.1149, DiabetesPedigreeFunction 0.0818, SkinThickness 0.0585,
Pregnancies 0.0571, BloodPressure 0.0473.

## Limitations

- Small, historical dataset (768 patients from one population) — results may
  not generalise to other populations.
- Nearly half the Insulin values were missing and had to be imputed, which
  weakens that feature.
- Precision is only ~0.61: many flagged patients would test negative, so this
  tool is a screening flag, never a diagnosis.
