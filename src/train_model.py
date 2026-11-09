"""
train_model.py
--------------
Loads the synthetic build dataset, trains a Random Forest Classifier,
evaluates it, and saves the model + evaluation artefacts.

Outputs:
  models/failure_model.pkl      – trained model (joblib format)
  results/evaluation.txt        – accuracy, classification report, confusion matrix
  results/feature_importance.png – bar chart of feature importances
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works on servers too
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# ---------------------------------------------------------------------------
# Paths – resolve relative to this script so it works from any working dir
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "ml", "builds.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH  = os.path.join(MODEL_DIR, "failure_model.pkl")
EVAL_PATH   = os.path.join(RESULTS_DIR, "evaluation.txt")
CHART_PATH  = os.path.join(RESULTS_DIR, "feature_importance.png")

# Create output directories if they don't exist
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at: {DATA_PATH}")
    print("  Please run:  python ml/generate_dataset.py  first.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("  AI-BASED BUILD-FAILURE PREDICTION – MODEL TRAINING")
print("=" * 60)
print(f"\n[1] Dataset loaded: {DATA_PATH}")
print(f"    Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# Step 2: Separate features (X) and target (y)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["files_changed", "lines_changed", "tests_touched", "hour"]
TARGET_COL   = "failed"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\n[2] Features : {FEATURE_COLS}")
print(f"    Target   : '{TARGET_COL}'  (0 = passed, 1 = failed)")
print(f"    Class distribution:\n{y.value_counts().to_string()}")

# ---------------------------------------------------------------------------
# Step 3: Train / test split  (80 % train, 20 % test)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n[3] Train/Test split (80/20)")
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# ---------------------------------------------------------------------------
# Step 4: Train Random Forest Classifier
# ---------------------------------------------------------------------------

clf = RandomForestClassifier(
    n_estimators=100,     # 100 decision trees
    max_depth=10,         # limit tree depth to avoid overfitting
    random_state=42,
    class_weight="balanced",  # handles slight class imbalance automatically
)

print("\n[4] Training RandomForestClassifier (100 trees) …")
clf.fit(X_train, y_train)
print("    Training complete.")

# ---------------------------------------------------------------------------
# Step 5: Evaluate the model
# ---------------------------------------------------------------------------

y_pred       = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=["Passed", "Failed"])
cm       = confusion_matrix(y_test, y_pred)

print("\n[5] Model Evaluation")
print(f"    Accuracy : {accuracy * 100:.2f} %")
print("\n    Classification Report:")
print(report)
print("    Confusion Matrix:")
print(f"    {cm}")

# ---------------------------------------------------------------------------
# Step 6: Feature importance
# ---------------------------------------------------------------------------

importances = clf.feature_importances_
feature_names = FEATURE_COLS

print("\n[6] Feature Importance:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"    {name:<20} {imp:.4f}")

# Save bar chart
fig, ax = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(importances)[::-1]
ax.bar(
    [feature_names[i] for i in sorted_idx],
    [importances[i]   for i in sorted_idx],
    color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
    edgecolor="black",
)
ax.set_title("Feature Importance – Build-Failure Prediction", fontsize=14, fontweight="bold")
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Importance Score", fontsize=12)
ax.set_ylim(0, 1)
for i, idx in enumerate(sorted_idx):
    ax.text(i, importances[idx] + 0.01, f"{importances[idx]:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=120)
plt.close()
print(f"\n    Feature importance chart saved → {CHART_PATH}")

# ---------------------------------------------------------------------------
# Step 7: Save evaluation text report
# ---------------------------------------------------------------------------

eval_text = f"""AI-BASED BUILD-FAILURE PREDICTION – EVALUATION REPORT
{"="*60}

Dataset       : {DATA_PATH}
Total records : {df.shape[0]}
Features      : {FEATURE_COLS}
Target        : {TARGET_COL}

Train samples : {len(X_train)}
Test  samples : {len(X_test)}

Model         : RandomForestClassifier (n_estimators=100, max_depth=10)

ACCURACY      : {accuracy * 100:.2f} %

CLASSIFICATION REPORT:
{report}

CONFUSION MATRIX:
{cm}
  Rows = Actual label   (0=Passed, 1=Failed)
  Cols = Predicted label(0=Passed, 1=Failed)

FEATURE IMPORTANCE:
"""
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    eval_text += f"  {name:<20} {imp:.4f}\n"

with open(EVAL_PATH, "w") as f:
    f.write(eval_text)
print(f"    Evaluation report saved  → {EVAL_PATH}")

# ---------------------------------------------------------------------------
# Step 8: Save trained model
# ---------------------------------------------------------------------------

joblib.dump(clf, MODEL_PATH)
print(f"\n[7] Model saved → {MODEL_PATH}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE – All artefacts saved successfully.")
print("=" * 60)

