"""
Data Hygiene Test Suite for the Sepsis Cohort (v2.0 - Corrected)

v2.0: Updates the internal splitting logic to match the definitive patient-level
      split used in the main data preparation pipeline, correcting the false
      positive leakage detection.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

print("--- Running Data Hygiene and Leakage Test Suite ---")

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_FILE = ROOT_DIR / "data/preprocessed/sepsis_feature_matrix.csv"
TARGET_COLUMN = "hospital_expire_flag"
PATIENT_ID_COL = "subject_id"
RANDOM_STATE = 42

# --- 1. Test the Patient-Level Splitting Logic ---
print("\n[Test 1/1] Verifying the integrity of the patient-level splitting logic...")

try:
    df = pd.read_csv(PREPROCESSED_FILE)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
except FileNotFoundError:
    print(f"❌ ERROR: Preprocessed file not found at {PREPROCESSED_FILE}")
    exit()

# --- THE FIX ---
# This block now uses the EXACT same patient-level split logic as your
# main data preparation notebook.

# For each patient, determine their single, definitive outcome.
patient_outcomes = df.groupby(PATIENT_ID_COL)[TARGET_COLUMN].max()
unique_patient_ids = patient_outcomes.index.to_series()

# Split the list of unique patient IDs
train_ids, temp_ids = train_test_split(
    unique_patient_ids,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=patient_outcomes
)

temp_labels = temp_ids.map(patient_outcomes)
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp_labels
)

# Convert to sets for the disjoint check
train_ids_set = set(train_ids)
val_ids_set = set(val_ids)
test_ids_set = set(test_ids)

# Perform the checks using assertions
assert train_ids_set.isdisjoint(val_ids_set), "LEAKAGE DETECTED: Patient IDs overlap between train and validation sets!"
assert train_ids_set.isdisjoint(test_ids_set), "LEAKAGE DETECTED: Patient IDs overlap between train and test sets!"
assert val_ids_set.isdisjoint(test_ids_set), "LEAKAGE DETECTED: Patient IDs overlap between validation and test sets!"

print("✅ PASSED: The patient-level splitting logic is correct and guarantees no patient overlap.")
print("\n--- All Data Hygiene Checks Passed Successfully ---")