"""
MIMIC-IV AKI Cohort Preprocessing (v3.0 - Final with Feature Selection)

This script builds the AKI cohort and generates the final feature matrix by:
1.  Extracting a comprehensive set of 104 clinically relevant features.
2.  Performing data-driven feature selection to keep only the top 40 most
    predictive features, plus age and gender, for the final model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# --- Constants & Configuration ---
# !!! UPDATE THIS TO MATCH YOUR LOCAL FOLDER STRUCTURE !!!
BASE_DATA_PATH = Path("D:/mimic-iv-3.1/mimic-iv-3.1/csv files/") 
HOSP_DATA_PATH = BASE_DATA_PATH / "hosp/"
ICU_DATA_PATH = BASE_DATA_PATH / "icu/"
OUTPUT_PATH = Path("data/preprocessed/")
TIME_WINDOW_HOURS = 24
AKI_ICD_CODES = ['N170', 'N171', 'N172', 'N178', 'N179', 'O904']
CREATININE_ITEMID = 50912

# --- Candidate Feature Sets (Used to generate 104 features for analysis) ---
VITAL_SIGN_ITEMIDS = {
    'heart_rate': 220045, 'sbp': 220179, 'dbp': 220180,
    'respiratory_rate': 220210, 'temperature_c': 223761, 'spo2': 220277
}
LAB_PANEL_ITEMIDS = {
    'hematocrit': 51221, 'hemoglobin': 51222, 'platelet': 51265, 'rbc': 51279,
    'wbc': 51301, 'bicarbonate': 50882, 'creatinine': 50912, 'glucose': 50931,
    'potassium': 50971, 'bun': 51006, 'aniongap': 50868, 'lactate': 50813,
    'ph': 50820, 'o2_saturation': 50817, 'base_excess': 50802, 'ptt': 51275,
    'inr': 51237, 'calcium': 50893, 'chloride': 50902, 'sodium': 50983,
    'pt': 51274, 'alt': 50861, 'alp': 50863, 'ast': 50878,
    'bilirubin_total': 50885, 'magnesium': 50960, 'phosphate': 50970, 'fibrinogen': 51214
}
ITEMID_TO_LAB_NAME = {v: k for k, v in LAB_PANEL_ITEMIDS.items()}
ITEMID_TO_VITAL_NAME = {v: k for k, v in VITAL_SIGN_ITEMIDS.items()}

def generate_kdigo_labels(cohort_icu_df, hosp_path):
    """
    Generates the binary AKI label based on KDIGO serum creatinine criteria.
    """
    print("Generating KDIGO ground truth labels...")
    creatinine_labs = pd.read_csv(hosp_path / 'labevents.csv', usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    creatinine_labs = creatinine_labs[
        creatinine_labs['hadm_id'].isin(cohort_icu_df['hadm_id'].unique()) & 
        (creatinine_labs['itemid'] == CREATININE_ITEMID)
    ].dropna(subset=['valuenum'])
    creatinine_labs['charttime'] = pd.to_datetime(creatinine_labs['charttime'])
    creatinine_labs = pd.merge(creatinine_labs, cohort_icu_df[['hadm_id', 'stay_id', 'intime']], on='hadm_id', how='left').sort_values(['stay_id', 'charttime'])
    rolling_min = creatinine_labs.groupby('stay_id').rolling(window='48h', on='charttime')['valuenum'].min().reset_index()
    merged_creat = pd.merge(creatinine_labs, rolling_min, on=['stay_id', 'charttime'], suffixes=('', '_rolling_min'))
    aki_by_creat_stage1 = merged_creat[merged_creat['valuenum'] >= merged_creat['valuenum_rolling_min'] + 0.3]['stay_id'].unique()
    seven_days_from_admission = (cohort_icu_df.groupby('stay_id')['intime'].min() + timedelta(days=7)).reset_index(name='baseline_endtime')
    baseline_df = pd.merge(creatinine_labs, seven_days_from_admission, on='stay_id')
    baseline_df = baseline_df[baseline_df['charttime'] <= baseline_df['baseline_endtime']]
    baseline_creat = baseline_df.groupby('stay_id')['valuenum'].min().reset_index(name='baseline_creat')
    creat_with_baseline = pd.merge(creatinine_labs, baseline_creat, on='stay_id', how='left')
    aki_by_creat_stage2 = creat_with_baseline[creat_with_baseline['valuenum'] >= 1.5 * creat_with_baseline['baseline_creat']]['stay_id'].unique()
    aki_stay_ids = set(aki_by_creat_stage1) | set(aki_by_creat_stage2)
    labeled_df = cohort_icu_df.copy()
    labeled_df['kdigo_aki'] = labeled_df['stay_id'].apply(lambda x: 1 if x in aki_stay_ids else 0)
    print(f"Labeling complete. Found {labeled_df['kdigo_aki'].sum()} AKI positive cases out of {len(labeled_df)} stays.")
    return labeled_df

def load_aki_cohort(hosp_path):
    """Loads admissions for patients with AKI-related ICD-10 codes and merges demographic data."""
    print("Loading AKI cohort...")
    diagnoses = pd.read_csv(hosp_path / 'diagnoses_icd.csv', usecols=['subject_id', 'icd_code'])
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
        chunk = chunk[chunk['stay_id'].isin(stay_windows_df['stay_id']) & chunk['itemid'].isin(VITAL_SIGN_ITEMIDS.values())]
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
    """Extracts and aggregates lab measurements from labevents using mean, min, and max."""
    print("Extracting labs from labevents (mean, min, max)...")
    relevant_hadm_ids = stay_windows_df['hadm_id'].unique()
    labs_all = []
    for chunk in pd.read_csv(hosp_path / 'labevents.csv', usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'], chunksize=10_000_000, low_memory=False):
        chunk.dropna(subset=['valuenum'], inplace=True)
        chunk = chunk[chunk['hadm_id'].isin(relevant_hadm_ids) & chunk['itemid'].isin(LAB_PANEL_ITEMIDS.values())]
        if not chunk.empty:
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            chunk_merged = pd.merge(chunk, stay_windows_df, on='hadm_id', how='left')
            labs_in_window = chunk_merged[(chunk_merged['charttime'] >= chunk_merged['intime']) & (chunk_merged['charttime'] <= chunk_merged['endtime'])]
            labs_all.append(labs_in_window[['stay_id', 'itemid', 'valuenum']])
    if not labs_all: return pd.DataFrame()
    labs_df = pd.concat(labs_all)
    labs_agg = labs_df.groupby(['stay_id', 'itemid'])['valuenum'].agg(['mean', 'min', 'max']).unstack()
    labs_agg.columns = [f"{ITEMID_TO_LAB_NAME[itemid]}_{stat}" for stat, itemid in labs_agg.columns]
    return labs_agg

def main():
    """Main execution pipeline for AKI preprocessing."""
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    
    aki_admissions = load_aki_cohort(HOSP_DATA_PATH)
    aki_icu = link_cohort_to_icu_stays(aki_admissions, ICU_DATA_PATH)
    aki_labeled = generate_kdigo_labels(aki_icu, HOSP_DATA_PATH)
    
    stay_windows = aki_labeled[['stay_id', 'hadm_id', 'intime', 'endtime']].copy()
    vitals_features = extract_vitals_for_stays(stay_windows, ICU_DATA_PATH)
    labs_features = extract_labs_for_stays(stay_windows, HOSP_DATA_PATH)
    
    print("Assembling comprehensive feature matrix for AKI cohort...")
    base_data = aki_labeled.set_index('stay_id')
    final_matrix = base_data.join(vitals_features).join(labs_features)
    final_matrix['gender'] = final_matrix['gender'].apply(lambda x: 1 if x == 'M' else 0)
    
    print("\nPerforming feature selection based on analysis...")
    
    static_cols_to_keep = ['subject_id', 'hadm_id', 'kdigo_aki', 'age', 'gender']
    top_40_features = [
        'creatinine_mean', 'creatinine_max', 'bun_mean', 'aniongap_mean', 'bicarbonate_mean',
        'creatinine_min', 'rbc_mean', 'hematocrit_mean', 'sbp_min', 'ph_mean',
        'heart_rate_max', 'hemoglobin_mean', 'respiratory_rate_mean', 'bun_max',
        'chloride_mean', 'sodium_mean', 'bun_min', 'ph_min', 'base_excess_mean',
        'wbc_mean', 'respiratory_rate_max', 'o2_saturation_min', 'spo2_min',
        'temperature_c_max', 'temperature_c_mean', 'heart_rate_mean', 'hematocrit_min',
        'phosphate_mean', 'ph_max', 'dbp_min', 'hemoglobin_min', 'ptt_mean',
        'calcium_mean', 'o2_saturation_mean', 'glucose_mean', 'platelet_mean',
        'magnesium_mean', 'base_excess_max', 'inr_mean', 'potassium_mean'
    ]
    
    # The final set of columns includes the IDs, target, and the selected features
    final_cols_to_keep = ['subject_id', 'hadm_id', 'stay_id'] + ['kdigo_aki', 'age', 'gender'] + top_40_features
    
    # We use .reindex() to ensure column order and handle any missing columns gracefully
    final_matrix_selected = final_matrix.reindex(columns=final_cols_to_keep)
    
    # Drop the stay_id from the final columns to be saved in the CSV file
    final_matrix_selected = final_matrix_selected.drop(columns=['stay_id'])
    
    output_file = OUTPUT_PATH / 'aki_feature_matrix.csv'
    final_matrix_selected.to_csv(output_file)
    
    print("\n--- AKI Preprocessing Complete ---")
    print(f"Final AKI matrix shape with selected features: {final_matrix_selected.shape}")
    print(f"Saved to: {output_file}")

if __name__ == '__main__':
    main()