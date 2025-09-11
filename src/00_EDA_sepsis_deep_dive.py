"""
Exploratory Data Analysis for the Sepsis Cohort.

This script performs two key functions:
1.  Analyzes and visualizes the mortality rate by Sepsis sub-category.
2.  Generates a formal JSON report on the missingness rate for each feature.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# --- Configuration ---
PREPROCESSED_FILE = "data/preprocessed/sepsis_feature_matrix.csv"
DIAGNOSES_FILE = "D:/mimic-iv-3.1/mimic-iv-3.1/csv files/hosp/diagnoses_icd.csv" # Update path if needed
REPORTS_DIR = Path("reports/")
TARGET_COLUMN = "hospital_expire_flag"
ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]

def generate_missingness_report(df, output_path):
    """Calculates the percentage of missing values for each feature and saves it as a JSON file."""
    print("\n--- Generating Missingness Report ---")
    
    feature_cols = df.drop(columns=[TARGET_COLUMN] + ID_COLUMNS, errors='ignore').columns
    missing_rates = df[feature_cols].isnull().mean().round(4) * 100
    missing_rates_dict = missing_rates.to_dict()
    
    # Save the report
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(missing_rates_dict, f, indent=4)
        
    print(f"✅ Missingness report saved to: {output_path}")
    
    # Also print the top 10 most missing features
    print("\nTop 10 most missing features (%):")
    print(missing_rates.sort_values(ascending=False).head(10))


def analyze_sepsis_subcategories(df, diagnoses_path):
    """Analyzes and plots mortality rates for the top 15 most frequent Sepsis sub-categories."""
    print("\n--- Analyzing Sepsis Sub-categories ---")
    
    # This requires the raw diagnoses file to link ICD codes
    try:
        diagnoses_df = pd.read_csv(diagnoses_path, usecols=['hadm_id', 'icd_code'])
    except FileNotFoundError:
        print(f"❌ ERROR: Raw diagnoses file not found at {diagnoses_path}")
        return

    # Merge to get ICD codes for each admission in our cohort
    df_with_icd = pd.merge(df, diagnoses_df, on='hadm_id')
    
    # Filter for Sepsis-related codes (optional, but good practice)
    sepsis_codes = df_with_icd[df_with_icd['icd_code'].str.startswith(('A40', 'A41', 'R65'), na=False)]
    
    # Get top 15 most frequent codes
    top_15_codes = sepsis_codes['icd_code'].value_counts().nlargest(15).index
    
    # Calculate mortality rate for these top codes
    mortality_by_code = sepsis_codes[sepsis_codes['icd_code'].isin(top_15_codes)].groupby('icd_code')[TARGET_COLUMN].mean().sort_values(ascending=False)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=mortality_by_code.values, y=mortality_by_code.index, palette='crest', ax=ax)
    ax.set_title('Mortality Rate by Sepsis Sub-Category (Top 15 Most Frequent)', fontsize=16)
    ax.set_xlabel('Mortality Rate', fontsize=12)
    ax.set_ylabel('Sepsis Sub-Category (ICD-10 Code)', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path("Images/") / "Mortality_Rate_by_Sepsis_Sub-Category_Top_15_Most_Frequent.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    print(f"✅ Mortality rate plot saved to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    try:
        sepsis_df = pd.read_csv(PREPROCESSED_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: Main preprocessed file not found at {PREPROCESSED_FILE}")
        exit()
        
    # Run both analysis functions
    generate_missingness_report(sepsis_df, REPORTS_DIR / "sepsis_missingness.json")
    analyze_sepsis_subcategories(sepsis_df, DIAGNOSES_FILE)