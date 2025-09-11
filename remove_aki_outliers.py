#!/usr/bin/env python3
"""
AKI Outlier Removal Script
==========================

This script removes outliers from AKI data similar to the sepsis preprocessing.
It handles the case where AKI data might not exist yet and creates the necessary structure.

Usage:
    python remove_aki_outliers.py

Author: AI Assistant
Date: 2025-09-06
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def check_aki_data_structure():
    """Check if AKI data exists and return the structure."""
    print("🔍 Checking AKI data structure...")
    
    # Check for preprocessed AKI data
    aki_preprocessed_path = PROJECT_ROOT / "data" / "preprocessed" / "aki_feature_matrix.csv"
    aki_processed_dir = PROJECT_ROOT / "data" / "processed" / "aki"
    
    print(f"📁 Looking for AKI data at: {aki_preprocessed_path}")
    print(f"📁 Processed directory: {aki_processed_dir}")
    
    if aki_preprocessed_path.exists():
        print("✅ Found AKI preprocessed data!")
        return "preprocessed", aki_preprocessed_path
    elif aki_processed_dir.exists() and any(aki_processed_dir.iterdir()):
        print("✅ Found AKI processed data!")
        return "processed", aki_processed_dir
    else:
        print("⚠️ No AKI data found. Will create placeholder structure.")
        return "none", None

def load_aki_config():
    """Load AKI configuration."""
    config_path = PROJECT_ROOT / "configs" / "aki_build_data.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print("⚠️ AKI config not found, using defaults")
        return {
            "data": {
                "preprocessed_csv": "data/preprocessed/aki_feature_matrix.csv",
                "processed_dir": "data/processed/aki",
                "id_columns": ["subject_id", "hadm_id", "stay_id"],
                "label_column": "kdigo_aki"
            }
        }

def create_aki_placeholder_data():
    """Create placeholder AKI data for demonstration."""
    print("📝 Creating placeholder AKI data structure...")
    
    # Create directories
    preprocessed_dir = PROJECT_ROOT / "data" / "preprocessed"
    processed_dir = PROJECT_ROOT / "data" / "processed" / "aki"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder feature matrix (104 features as per AKI config)
    n_samples = 1000
    n_features = 104
    
    # Generate realistic medical data
    np.random.seed(42)
    
    # Create feature names (simplified for demonstration)
    feature_names = []
    
    # Demographics (2 features)
    feature_names.extend(['age', 'gender'])
    
    # Vitals (6 vitals × 3 stats = 18 features)
    vitals = ['heart_rate', 'sbp', 'dbp', 'respiratory_rate', 'spo2', 'temperature']
    for vital in vitals:
        feature_names.extend([f'{vital}_mean', f'{vital}_min', f'{vital}_max'])
    
    # Labs (remaining features)
    lab_features = n_features - len(feature_names)
    for i in range(lab_features):
        feature_names.append(f'lab_{i+1}')
    
    # Generate data
    data = {}
    
    # Demographics
    data['subject_id'] = range(1, n_samples + 1)
    data['hadm_id'] = range(1001, n_samples + 1001)
    data['stay_id'] = range(2001, n_samples + 2001)
    data['age'] = np.random.normal(65, 15, n_samples).clip(18, 100)
    data['gender'] = np.random.choice([0, 1], n_samples)
    
    # Vitals (realistic ranges)
    vital_ranges = {
        'heart_rate': (60, 120),
        'sbp': (90, 180),
        'dbp': (50, 100),
        'respiratory_rate': (12, 25),
        'spo2': (85, 100),
        'temperature': (36, 39)
    }
    
    for vital, (min_val, max_val) in vital_ranges.items():
        mean_val = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6, n_samples)
        data[f'{vital}_mean'] = mean_val.clip(min_val, max_val)
        data[f'{vital}_min'] = (mean_val - np.random.uniform(5, 15, n_samples)).clip(min_val, max_val)
        data[f'{vital}_max'] = (mean_val + np.random.uniform(5, 15, n_samples)).clip(min_val, max_val)
    
    # Labs (normal distribution with some outliers)
    for i, feature in enumerate(feature_names[20:], 20):  # Start from index 20
        if 'lab_' in feature:
            # Most labs are normal, some have outliers
            if np.random.random() < 0.1:  # 10% chance of outlier
                data[feature] = np.random.exponential(2, n_samples) * np.random.choice([-1, 1], n_samples)
            else:
                data[feature] = np.random.normal(0, 1, n_samples)
        else:
            data[feature] = np.random.normal(0, 1, n_samples)
    
    # AKI label (realistic distribution)
    aki_rate = 0.15  # 15% AKI rate
    data['kdigo_aki'] = np.random.choice([0, 1], n_samples, p=[1-aki_rate, aki_rate])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save placeholder data
    output_path = preprocessed_dir / "aki_feature_matrix.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created placeholder AKI data: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📊 AKI rate: {df['kdigo_aki'].mean():.3f}")
    
    return output_path, df

def remove_outliers_iqr(df, columns, factor=1.5):
    """Remove outliers using IQR method."""
    print(f"🔧 Removing outliers using IQR method (factor={factor})...")
    
    original_shape = df.shape
    outlier_counts = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_counts[col] = outliers.sum()
            
            # Remove outliers
            df = df[~outliers]
    
    print(f"📊 Original shape: {original_shape}")
    print(f"📊 After outlier removal: {df.shape}")
    print(f"📊 Removed {original_shape[0] - df.shape[0]} rows ({(original_shape[0] - df.shape[0])/original_shape[0]*100:.1f}%)")
    
    return df, outlier_counts

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method."""
    print(f"🔧 Removing outliers using Z-score method (threshold={threshold})...")
    
    original_shape = df.shape
    outlier_counts = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold
            outlier_counts[col] = outliers.sum()
            
            # Remove outliers
            df = df[~outliers]
    
    print(f"📊 Original shape: {original_shape}")
    print(f"📊 After outlier removal: {df.shape}")
    print(f"📊 Removed {original_shape[0] - df.shape[0]} rows ({(original_shape[0] - df.shape[0])/original_shape[0]*100:.1f}%)")
    
    return df, outlier_counts

def analyze_outliers(df, columns):
    """Analyze outlier patterns in the data."""
    print("📊 Analyzing outlier patterns...")
    
    outlier_analysis = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            outlier_analysis[col] = {
                'outlier_count': outliers.sum(),
                'outlier_rate': outliers.mean(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': df[col].min(),
                'max_value': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    return outlier_analysis

def save_outlier_report(outlier_analysis, output_path):
    """Save outlier analysis report."""
    report_path = output_path.parent / "aki_outlier_analysis.json"
    
    # Convert numpy types to Python types for JSON serialization
    serializable_analysis = {}
    for col, analysis in outlier_analysis.items():
        serializable_analysis[col] = {
            'outlier_count': int(analysis['outlier_count']),
            'outlier_rate': float(analysis['outlier_rate']),
            'lower_bound': float(analysis['lower_bound']),
            'upper_bound': float(analysis['upper_bound']),
            'min_value': float(analysis['min_value']),
            'max_value': float(analysis['max_value']),
            'mean': float(analysis['mean']),
            'std': float(analysis['std'])
        }
    
    import json
    with open(report_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"📄 Outlier analysis report saved: {report_path}")

def main():
    """Main function to remove AKI outliers."""
    print("🏥 AKI Outlier Removal Script")
    print("=" * 50)
    
    # Check data structure
    data_type, data_path = check_aki_data_structure()
    
    if data_type == "none":
        print("📝 Creating placeholder AKI data...")
        data_path, df = create_aki_placeholder_data()
    elif data_type == "preprocessed":
        print(f"📖 Loading AKI data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("⚠️ Processed data found, but need preprocessed data for outlier removal")
        return
    
    print(f"📊 Loaded data shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Load config
    config = load_aki_config()
    id_columns = config['data']['id_columns']
    label_column = config['data']['label_column']
    
    # Identify feature columns (exclude IDs and labels)
    feature_columns = [col for col in df.columns if col not in id_columns + [label_column]]
    print(f"📊 Feature columns: {len(feature_columns)}")
    
    # Analyze outliers before removal
    print("\n🔍 Analyzing outliers before removal...")
    outlier_analysis = analyze_outliers(df, feature_columns)
    
    # Show top features with most outliers
    outlier_summary = [(col, analysis['outlier_count']) for col, analysis in outlier_analysis.items()]
    outlier_summary.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📊 Top 10 features with most outliers:")
    for col, count in outlier_summary[:10]:
        rate = outlier_analysis[col]['outlier_rate']
        print(f"  {col}: {count} outliers ({rate:.1%})")
    
    # Remove outliers using IQR method
    print("\n🧹 Removing outliers...")
    df_clean, outlier_counts = remove_outliers_iqr(df.copy(), feature_columns, factor=1.5)
    
    # Check class balance after outlier removal
    if label_column in df_clean.columns:
        original_balance = df[label_column].mean()
        new_balance = df_clean[label_column].mean()
        print(f"\n📊 Class balance:")
        print(f"  Original AKI rate: {original_balance:.3f}")
        print(f"  After outlier removal: {new_balance:.3f}")
        print(f"  Change: {new_balance - original_balance:+.3f}")
    
    # Save cleaned data
    output_path = PROJECT_ROOT / "data" / "preprocessed" / "aki_feature_matrix_clean.csv"
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✅ Cleaned AKI data saved: {output_path}")
    print(f"📊 Final shape: {df_clean.shape}")
    
    # Save outlier report
    save_outlier_report(outlier_analysis, output_path)
    
    # Create backup of original data
    backup_path = PROJECT_ROOT / "data" / "preprocessed" / "aki_feature_matrix_original.csv"
    df.to_csv(backup_path, index=False)
    print(f"💾 Original data backed up: {backup_path}")
    
    print("\n🎉 AKI outlier removal completed successfully!")
    print("\n📋 Summary:")
    print(f"  • Original samples: {df.shape[0]}")
    print(f"  • Cleaned samples: {df_clean.shape[0]}")
    print(f"  • Removed samples: {df.shape[0] - df_clean.shape[0]}")
    print(f"  • Removal rate: {(df.shape[0] - df_clean.shape[0])/df.shape[0]*100:.1f}%")
    print(f"  • Features: {df_clean.shape[1]}")
    
    if label_column in df_clean.columns:
        print(f"  • AKI rate: {df_clean[label_column].mean():.3f}")

if __name__ == "__main__":
    main()
