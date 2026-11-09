"""
generate_dataset.py
-------------------
Generates 300 synthetic build records for the Build-Failure Prediction project.

NOTE: This dataset is entirely synthetic/simulated for demonstration purposes.
      It does NOT represent real company production data.

Features:
  - files_changed  : Number of source files modified in the commit
  - lines_changed  : Total lines of code added/removed
  - tests_touched  : Number of test files modified
  - hour           : Hour of the day when the build was triggered (0-23)

Target:
  - failed         : 1 = build failed, 0 = build succeeded
"""

import numpy as np
import pandas as pd
import os

# Fix the random seed so results are reproducible every run
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

NUM_RECORDS = 300

# ---------------------------------------------------------------------------
# Step 1: Generate raw feature values
# ---------------------------------------------------------------------------

files_changed = rng.integers(1, 25, size=NUM_RECORDS)          # 1–24 files
lines_changed = rng.integers(5, 600, size=NUM_RECORDS)         # 5–599 lines
tests_touched = rng.integers(0, 15, size=NUM_RECORDS)          # 0–14 test files
hour          = rng.integers(0, 24, size=NUM_RECORDS)          # 0–23

# ---------------------------------------------------------------------------
# Step 2: Assign failure probability based on realistic patterns
#
# Higher failure risk when:
#   - Many files are changed (large diff)
#   - Many lines are changed (large diff)
#   - Few or no tests are touched (no test coverage for the change)
#   - Build runs late at night or very early morning (less careful commits)
# ---------------------------------------------------------------------------

# Normalise each feature to [0, 1] so we can combine them
files_norm  = (files_changed - 1)  / (24 - 1)
lines_norm  = (lines_changed - 5)  / (600 - 5)
tests_norm  = (tests_touched)      / 14          # more tests → lower risk
hour_norm   = np.abs(hour - 12)    / 12          # peak risk at hour 0 or 23

# Weighted combination → raw failure score
failure_score = (
    0.35 * files_norm
    + 0.35 * lines_norm
    - 0.20 * tests_norm   # more tests REDUCES failure probability
    + 0.10 * hour_norm
)

# Clip to [0, 1] and convert to actual 0/1 label with some noise
failure_prob = np.clip(failure_score, 0, 1)

# Add random noise to avoid perfectly deterministic labels
noise = rng.uniform(-0.08, 0.08, size=NUM_RECORDS)
failure_prob = np.clip(failure_prob + noise, 0, 1)

# Convert probability to binary label using a lower threshold → ~45% failure rate
failed = (failure_prob > 0.35).astype(int)

# ---------------------------------------------------------------------------
# Step 3: Assemble DataFrame and save as CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    "files_changed": files_changed,
    "lines_changed":  lines_changed,
    "tests_touched":  tests_touched,
    "hour":           hour,
    "failed":         failed,
})

# Make sure the ml/ directory exists (it always does, but just in case)
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builds.csv")
df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------------
# Step 4: Print a summary
# ---------------------------------------------------------------------------

print("=" * 55)
print("  BUILD DATASET GENERATION COMPLETE")
print("=" * 55)
print(f"  Total records  : {len(df)}")
print(f"  Failed builds  : {failed.sum()}  ({failed.mean()*100:.1f} %)")
print(f"  Passed builds  : {(1-failed).sum()}  ({(1-failed.mean())*100:.1f} %)")
print(f"  Saved to       : {output_path}")
print("=" * 55)
print("\nFirst 5 rows of the dataset:")
print(df.head().to_string(index=False))
print("\n[NOTE] This is SYNTHETIC data for demonstration only.")
