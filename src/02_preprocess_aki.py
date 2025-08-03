"""
MIMIC-IV AKI Cohort Preprocessing with KDIGO Label Generation (Fast Version)

Description:
This script builds the Acute Kidney Injury (AKI) cohort and engineers the 
'kdigo_aki' label by implementing the KDIGO serum creatinine criteria. This version 
is optimized for speed on machines with sufficient RAM (>16 GB) by processing
all patients at once.

"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# --- Constants & Configuration ---

# Paths for a local system.
# !!! UPDATE THESE TO MATCH YOUR LOCAL FOLDER STRUCTURE !!!
# HOSP_DATA_PATH = Path("../../mimic_iv_data/hosp/")
# ICU_DATA_PATH = Path("../../mimic_iv_data/icu/")
# OUTPUT_PATH = Path("../data/processed/")

BASE_DATA_PATH = Path("D:/mimic-iv-3.1/mimic-iv-3.1/csv files/")
HOSP_DATA_PATH = BASE_DATA_PATH / "hosp/"
ICU_DATA_PATH = BASE_DATA_PATH / "icu/"

# Define where the output file will be saved in your Drive.
OUTPUT_PATH = Path("data/processed/")   

# Time window for feature extraction post-ICU admission.
TIME_WINDOW_HOURS = 24

# Clinical variables finalized from the project's data analysis phase.
AKI_ICD_CODES = ['N170', 'N171', 'N172', 'N178', 'N179', 'O904']

# Item ID for Serum Creatinine, essential for KDIGO criteria.
CREATININE_ITEMID = 50912

# Feature sets, consistent with the Sepsis cohort for modeling purposes.
LAB_PANEL_ITEMIDS = {
    'cbc_hematocrit': 51221, 'cbc_hemoglobin': 51222, 'cbc_platelet': 51265,
    'cbc_rbc': 51279, 'cbc_wbc': 51301, 'cmp_bicarbonate': 50882,
    'cmp_creatinine': 50912, 'cmp_glucose': 50931, 'cmp_potassium': 50971,
    'cmp_bun': 51006, 'cmp_aniongap': 50868, 'cmp_lactate': 50813,
    'abg_ph': 50820, 'abg_o2_saturation': 50817, 'abg_base_excess': 50802,
    'aptt_ptt': 51275, 'aptt_inr': 51237
}

VITAL_SIGN_ITEMIDS = {
    'heart_rate': 220045, 'sbp': 220179, 'dbp': 220180,
    'respiratory_rate': 220210, 'temperature_c': 223761, 'spo2': 220277
}

# Invert dictionaries for efficient name lookup during column renaming.
ITEMID_TO_LAB_NAME = {v: k for k, v in LAB_PANEL_ITEMIDS.items()}
ITEMID_TO_VITAL_NAME = {v: k for k, v in VITAL_SIGN_ITEMIDS.items()}


def generate_kdigo_labels(cohort_icu_df, hosp_path):
    """
    Generates the binary AKI label based on KDIGO serum creatinine criteria.
    
    A patient is labeled positive (1) if their record meets either a time-
    sensitive (>=0.3 mg/dL increase in 48h) or baseline-relative 
    (>=1.5x baseline) increase in creatinine during their ICU stay.
    """
    print("Generating KDIGO ground truth labels...")
    
    # Isolate all creatinine values for the cohort's hospital admissions.
    creatinine_labs = pd.read_csv(hosp_path / 'labevents.csv', usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    creatinine_labs = creatinine_labs[
        creatinine_labs['hadm_id'].isin(cohort_icu_df['hadm_id'].unique()) & 
        (creatinine_labs['itemid'] == CREATININE_ITEMID)
    ].dropna(subset=['valuenum'])
    creatinine_labs['charttime'] = pd.to_datetime(creatinine_labs['charttime'])
    
    # Map hospital admissions to specific ICU stays.
    creatinine_labs = pd.merge(creatinine_labs, cohort_icu_df[['hadm_id', 'stay_id', 'intime']], on='hadm_id', how='left')
    creatinine_labs = creatinine_labs.sort_values(['stay_id', 'charttime'])

    # --- KDIGO Criterion 1: Increase of >= 0.3 mg/dL within 48 hours ---
    rolling_min = creatinine_labs.groupby('stay_id').rolling(window='48h', on='charttime')['valuenum'].min().reset_index()
    merged_creat = pd.merge(creatinine_labs, rolling_min, on=['stay_id', 'charttime'], suffixes=('', '_rolling_min'))
    aki_by_creat_stage1 = merged_creat[merged_creat['valuenum'] >= merged_creat['valuenum_rolling_min'] + 0.3]['stay_id'].unique()

    # --- KDIGO Criterion 2: Increase of >= 1.5x baseline ---
    # Baseline is defined as the minimum creatinine in the first 7 days of an ICU stay.
    seven_days_from_admission = cohort_icu_df.groupby('stay_id')['intime'].min() + timedelta(days=7)
    seven_days_from_admission = seven_days_from_admission.reset_index(name='baseline_endtime')
    
    baseline_df = pd.merge(creatinine_labs, seven_days_from_admission, on='stay_id')
    baseline_df = baseline_df[baseline_df['charttime'] <= baseline_df['baseline_endtime']]
    baseline_creat = baseline_df.groupby('stay_id')['valuenum'].min().reset_index(name='baseline_creat')
    
    creat_with_baseline = pd.merge(creatinine_labs, baseline_creat, on='stay_id', how='left')
    aki_by_creat_stage2 = creat_with_baseline[creat_with_baseline['valuenum'] >= 1.5 * creat_with_baseline['baseline_creat']]['stay_id'].unique()

    # Union of stays meeting either criterion forms the final positive set.
    aki_stay_ids = set(aki_by_creat_stage1) | set(aki_by_creat_stage2)
    
    labeled_df = cohort_icu_df.copy()
    labeled_df['kdigo_aki'] = labeled_df['stay_id'].apply(lambda x: 1 if x in aki_stay_ids else 0)
    
    print(f"Labeling complete. Found {labeled_df['kdigo_aki'].sum()} AKI positive cases out of {len(labeled_df)} stays.")
    return labeled_df


def load_aki_cohort(hosp_path):
    """Loads admissions for patients with AKI-related ICD-10 codes and merges demographic data."""
    print("Loading AKI cohort...")
    diagnoses = pd.read_csv(hosp_path / 'diagnoses_icd.csv')
    aki_subjects = diagnoses[diagnoses['icd_code'].isin(AKI_ICD_CODES)]['subject_id'].unique()
    
    patients = pd.read_csv(hosp_path / 'patients.csv', usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    admissions = pd.read_csv(hosp_path / 'admissions.csv', usecols=['subject_id', 'hadm_id', 'admittime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    
    aki_admissions = admissions[admissions['subject_id'].isin(aki_subjects)].copy()
    aki_admissions = pd.merge(aki_admissions, patients, on='subject_id')
    aki_admissions['age'] = aki_admissions['anchor_age'] + (aki_admissions['admittime'].dt.year - aki_admissions['anchor_year'])
    
    print(f"Identified {len(aki_subjects)} unique patients and {len(aki_admissions)} AKI-related admissions.")
    return aki_admissions

def link_cohort_to_icu_stays(cohort_df, icu_path):
    """Links hospital admissions to ICU stays and defines the 24h feature window."""
    print("Linking cohort to ICU stays...")
    icustays = pd.read_csv(icu_path / 'icustays.csv', usecols=['hadm_id', 'stay_id', 'intime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    cohort_icu = pd.merge(cohort_df, icustays, on='hadm_id')
    cohort_icu['endtime'] = cohort_icu['intime'] + timedelta(hours=TIME_WINDOW_HOURS)
    
    return cohort_icu.drop_duplicates(subset=['hadm_id', 'stay_id'])

def extract_vitals_for_stays(stay_windows_df, icu_path):
    """Extracts and aggregates vital signs from chartevents."""
    print("Extracting vitals from chartevents...")
    vitals_all = []
    for chunk in pd.read_csv(icu_path / 'chartevents.csv', usecols=['stay_id', 'itemid', 'charttime', 'valuenum'], chunksize=10_000_000, low_memory=False):
        chunk.dropna(subset=['valuenum'], inplace=True)
        chunk = chunk[chunk['stay_id'].isin(stay_windows_df['stay_id'])]
        chunk = chunk[chunk['itemid'].isin(VITAL_SIGN_ITEMIDS.values())]
        if not chunk.empty:
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            chunk_merged = pd.merge(chunk, stay_windows_df, on='stay_id', how='left')
            vitals_in_window = chunk_merged[(chunk_merged['charttime'] >= chunk_merged['intime']) & (chunk_merged['charttime'] <= chunk_merged['endtime'])]
            vitals_all.append(vitals_in_window[['stay_id', 'itemid', 'valuenum']])
    if not vitals_all: return pd.DataFrame()
    vitals_df = pd.concat(vitals_all)
    vitals_agg = vitals_df.groupby(['stay_id', 'itemid'])['valuenum'].agg(['mean', 'min', 'max']).unstack()
    vitals_agg.columns = [f"{ITEMID_TO_VITAL_NAME[itemid]}_{stat}" for stat, itemid in vitals_agg.columns]
    return vitals_agg

def extract_labs_for_stays(stay_windows_df, hosp_path):
    """Extracts and aggregates lab measurements from labevents."""
    print("Extracting labs from labevents...")
    relevant_hadm_ids = stay_windows_df['hadm_id'].unique()
    labs_all = []
    for chunk in pd.read_csv(hosp_path / 'labevents.csv', usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'], chunksize=10_000_000, low_memory=False):
        chunk.dropna(subset=['valuenum'], inplace=True)
        chunk = chunk[chunk['hadm_id'].isin(relevant_hadm_ids)]
        chunk = chunk[chunk['itemid'].isin(LAB_PANEL_ITEMIDS.values())]
        if not chunk.empty:
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            chunk_merged = pd.merge(chunk, stay_windows_df, on='hadm_id', how='left')
            labs_in_window = chunk_merged[(chunk_merged['charttime'] >= chunk_merged['intime']) & (chunk_merged['charttime'] <= chunk_merged['endtime'])]
            labs_all.append(labs_in_window[['stay_id', 'itemid', 'valuenum']])
    if not labs_all: return pd.DataFrame()
    labs_df = pd.concat(labs_all)
    labs_agg = labs_df.groupby(['stay_id', 'itemid'])['valuenum'].mean().unstack()
    labs_agg.columns = [ITEMID_TO_LAB_NAME[itemid] for itemid in labs_agg.columns]
    return labs_agg

def main():
    """Main execution pipeline for AKI preprocessing."""
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # 1. Load base AKI cohort using ICD codes.
    aki_admissions = load_aki_cohort(HOSP_DATA_PATH)
    
    # 2. Link hospital admissions to ICU stays.
    aki_icu = link_cohort_to_icu_stays(aki_admissions, ICU_DATA_PATH)

    # 3. Generate the KDIGO ground truth label for each ICU stay.
    aki_labeled = generate_kdigo_labels(aki_icu, HOSP_DATA_PATH)
    
    # 4. Extract time-series features from the first 24 hours.
    stay_windows = aki_labeled[['stay_id', 'hadm_id', 'intime', 'endtime']].copy()
    vitals_features = extract_vitals_for_stays(stay_windows, ICU_DATA_PATH)
    labs_features = extract_labs_for_stays(stay_windows, HOSP_DATA_PATH)

    # 5. Assemble the final feature matrix.
    print("Assembling final feature matrix for AKI cohort...")
    base_data = aki_labeled.set_index('stay_id')
    final_matrix = base_data.join(vitals_features).join(labs_features)
    final_matrix['gender'] = final_matrix['gender'].apply(lambda x: 1 if x == 'M' else 0)
    
    static_cols = ['subject_id', 'hadm_id', 'age', 'gender', 'kdigo_aki']
    feature_cols = list(vitals_features.columns) + list(labs_features.columns)
    
    # Ensure all feature columns exist, filling with NaN if a feature was never recorded for any patient.
    for col in feature_cols:
        if col not in final_matrix:
            final_matrix[col] = np.nan
            
    final_matrix = final_matrix[static_cols + feature_cols]

    # 6. Save the final matrix to the processed data directory.
    output_file = OUTPUT_PATH / 'aki_feature_matrix.csv'
    final_matrix.to_csv(output_file)
    
    print("\n--- AKI Preprocessing Complete ---")
    print(f"Final AKI matrix shape: {final_matrix.shape}")
    print(f"Saved to: {output_file}")


if __name__ == '__main__':
    main()