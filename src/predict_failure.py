"""
predict_failure.py
------------------
Loads the trained Random Forest model and predicts build-failure probability
for two example builds.

Risk thresholds:
  > 85 %  → High Risk     – Recommend additional testing and code review
  > 50 %  → Medium Risk   – Warn the developer
  < 15 %  → Low Risk      – Eligible for fast-path review
  else    → Moderate Risk – Normal CI checks

DISCLAIMER: Predictions are experimental and based on synthetic training data.
            Do NOT use this tool for critical production decisions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "failure_model.pkl")

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    print("  Please run:  python src/train_model.py  first.")
    sys.exit(1)

clf = joblib.load(MODEL_PATH)
FEATURE_COLS = ["files_changed", "lines_changed", "tests_touched", "hour"]

# ---------------------------------------------------------------------------
# Helper function: predict and display result for a single build
# ---------------------------------------------------------------------------

def classify_risk(probability_percent: float) -> tuple[str, str]:
    """
    Map a failure probability (0–100) to a risk category and suggested action.
    Returns (risk_label, suggested_action).
    """
    if probability_percent > 85:
        return (
            "🔴 HIGH RISK",
            "Strongly recommend additional testing and a thorough code review "
            "before merging. Consider blocking the merge until issues are resolved."
        )
    elif probability_percent > 50:
        return (
            "🟠 MEDIUM RISK",
            "Warning: elevated failure probability. Developer should double-check "
            "the changes, run all relevant tests, and get a peer review."
        )
    elif probability_percent < 15:
        return (
            "🟢 LOW RISK",
            "Build looks healthy. This commit is eligible for a fast-path review. "
            "Normal CI checks are sufficient."
        )
    else:
        return (
            "🟡 MODERATE RISK",
            "Proceed with standard CI checks. Ensure all existing tests pass "
            "before merging."
        )


def predict_build(build_name: str, features: dict) -> None:
    """
    Given a dictionary of feature values, predict the build outcome and print
    a formatted report.
    """
    # Build a single-row DataFrame (preserves feature names → no sklearn warning)
    input_df = pd.DataFrame([features], columns=FEATURE_COLS)

    prediction       = clf.predict(input_df)[0]          # 0 or 1
    failure_prob_raw = clf.predict_proba(input_df)[0][1]  # probability of class 1
    failure_pct      = failure_prob_raw * 100

    outcome      = "FAIL ❌" if prediction == 1 else "PASS ✅"
    risk, action = classify_risk(failure_pct)

    print(f"\n{'─'*55}")
    print(f"  Build Example : {build_name}")
    print(f"{'─'*55}")
    print(f"  Input Features:")
    for key, val in features.items():
        print(f"    {key:<20} : {val}")
    print(f"\n  Predicted Outcome    : {outcome}")
    print(f"  Failure Probability  : {failure_pct:.1f} %")
    print(f"  Risk Category        : {risk}")
    print(f"\n  Suggested Action:")
    print(f"    {action}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Example A – High-risk build
# ---------------------------------------------------------------------------

example_a = {
    "files_changed": 12,
    "lines_changed": 450,
    "tests_touched": 0,
    "hour":          22,
}

# Example B – Low-risk build
example_b = {
    "files_changed": 2,
    "lines_changed": 20,
    "tests_touched": 5,
    "hour":          10,
}

# ---------------------------------------------------------------------------
# Main output
# ---------------------------------------------------------------------------

print("=" * 55)
print("  AI-BASED BUILD-FAILURE PREDICTION")
print("=" * 55)
print(f"\n  Model loaded from : {MODEL_PATH}")
print("\n  ⚠️  DISCLAIMER: Predictions are experimental and are")
print("      based solely on synthetic training data.")
print("      Do not use for real production decisions.\n")

predict_build("Example A – High-Risk Build", example_a)
predict_build("Example B – Low-Risk  Build", example_b)

print("\n" + "=" * 55)
print("  PREDICTION COMPLETE")
print("=" * 55)

