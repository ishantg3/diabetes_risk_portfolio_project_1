"""Diabetes Risk Predictor — Streamlit frontend.

No binary model file is used. On startup the app loads the cleaned dataset
(data/diabetes_cleaned.csv, next to this file) and retrains the chosen best
pipeline — a tuned Random Forest — on the full cleaned data. The dataset is
tiny (768 rows), so training takes only a few seconds.

Chosen model (from notebooks/diabetes_model.ipynb):
    RandomForestClassifier(n_estimators=200, max_depth=5,
                           min_samples_split=5, class_weight="balanced",
                           random_state=42)
inside a Pipeline with a median imputer.

Educational demo only — NOT a medical diagnosis.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------- constants
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes_cleaned.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Slider ranges: (min, max, step) — based on the observed data ranges.
RANGES = {
    "Pregnancies": (0, 17, 1),
    "Glucose": (0, 200, 1),
    "BloodPressure": (0, 122, 1),
    "SkinThickness": (0, 99, 1),
    "Insulin": (0, 846, 1),
    "BMI": (0.0, 67.1, 0.1),
    "DiabetesPedigreeFunction": (0.08, 2.42, 0.01),
    "Age": (21, 81, 1),
}

# Test-set metrics of the chosen tuned Random Forest, measured in the notebook
# on the held-out stratified 20% test set (154 patients, 54 diabetic).
TEST_METRICS = {
    "Accuracy": 0.7468,
    "Precision": 0.6056,
    "Recall": 0.7963,
    "F1": 0.6880,
    "ROC-AUC": 0.8185,
}
CONFUSION = {"TN": 72, "FP": 28, "FN": 11, "TP": 43}


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Training the diabetes risk model...")
def load_model():
    """Load the cleaned data and retrain the chosen pipeline on all of it."""
    df = pd.read_csv(DATA_PATH)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(df[FEATURES], df["Outcome"])
    return pipeline, df


pipeline, df = load_model()
medians = df[FEATURES].median()

# ---------------------------------------------------------------- UI
st.title("Diabetes Risk Predictor")
st.write(
    "Enter the patient's measurements below and press **Predict**. "
    "Defaults are the median values from the training data, so you can "
    "change just one or two values and see how the prediction moves."
)

st.subheader("Patient measurements")
inputs = {}
for feature in FEATURES:
    lo, hi, step = RANGES[feature]
    default = float(medians[feature])
    default = max(lo, min(hi, default))
    # Streamlit requires value, min_value, max_value and step to share a
    # type: integer sliders get ints, decimal sliders get floats.
    if (float(lo).is_integer() and float(hi).is_integer()
            and float(step).is_integer()):
        lo, hi, step, default = int(lo), int(hi), int(step), int(round(default))
    else:
        lo, hi, step, default = float(lo), float(hi), float(step), float(default)
    inputs[feature] = st.slider(feature, min_value=lo, max_value=hi,
                                value=default, step=step)

if st.button("Predict"):
    row = pd.DataFrame([inputs], columns=FEATURES)
    prediction = int(pipeline.predict(row)[0])
    probability = float(pipeline.predict_proba(row)[0][1])

    st.subheader("Result")
    if prediction == 1:
        st.error("Prediction: Diabetic")
        st.write(
            f"The model estimates a **{probability:.1%} probability** of diabetes. "
            "This is a high-risk flag: the patient should be referred for proper "
            "clinical testing (for example, a fasting glucose or HbA1c test) — "
            "this app alone cannot diagnose anyone."
        )
    else:
        st.success("Prediction: Not Diabetic")
        st.write(
            f"The model estimates a **{probability:.1%} probability** of diabetes. "
            "That is below the model's decision threshold, but it is not a clean "
            "bill of health: if symptoms or risk factors are present, clinical "
            "testing is still the right next step."
        )
    st.metric("Predicted probability of diabetes", f"{probability:.1%}")

# ---------------------------------------------------------------- model info
st.subheader("About this model")
st.write(
    "Best model: **tuned Random Forest** "
    "(200 trees, max depth 5, min samples split 5, class_weight='balanced'), "
    "chosen by test F1 after two modelling iterations in "
    "`notebooks/diabetes_model.ipynb`. Iteration 2 beat Iteration 1 by "
    "**+0.0480 F1** (0.6400 → 0.6880), with Recall rising from 0.5926 to 0.7963 — "
    "the tuned model misses far fewer diabetic patients."
)
st.write("Test-set metrics (154 held-out patients, 54 diabetic):")
st.table(pd.DataFrame([TEST_METRICS]).round(4))
st.write(
    "Confusion matrix on the test set — "
    f"True Negatives {CONFUSION['TN']}, False Positives {CONFUSION['FP']}, "
    f"False Negatives {CONFUSION['FN']}, True Positives {CONFUSION['TP']}. "
    f"Of 54 actually diabetic patients, {CONFUSION['FN']} were missed."
)

importances = pd.Series(
    pipeline.named_steps["model"].feature_importances_, index=FEATURES
).sort_values(ascending=False)
st.write("What drives the prediction most (feature importances, retrained model):")
st.bar_chart(importances)

st.warning(
    "Disclaimer: this is an **educational demo, not medical advice and not a "
    "medical diagnosis**. Predictions come from a model trained on 768 historical "
    "records and can be wrong. Always consult a qualified healthcare professional "
    "for screening, diagnosis, or treatment decisions."
)
