#!/usr/bin/env python3
"""
Sepsis Project Cleanup Script
Removes unnecessary files and organizes the project structure
"""

import os
import shutil
from pathlib import Path

def cleanup_sepsis_project():
    """Clean up the Sepsis project by removing unnecessary files"""
    
    print("🧹 Starting Sepsis Project Cleanup...")
    
    # Files to delete (duplicate/outdated Sepsis models)
    sepsis_models_to_delete = [
        "models/rl_agent_sepsis_fast_uf-1_5_2diag.zip",
        "models/rl_agent_sepsis_recallboost_v1.zip",
        "models/rl_agent_sepsis_single_acc.zip",
        "models/rl_agent_sepsis_accfocus_final_v1.zip",
        "models/rl_agent_sepsis_interim_main.zip",
        "models/rl_agent_sepsis_asymmetric_final.zip",
        "models/rl_agent_sepsis_ablation_no_gain.zip",
        "models/rl_agent_sepsis_ablation_no_masking.zip",
        "models/rl_agent_sepsis_ablation_no_transformer.zip",
        "models/rl_agent_sepsis_masked_final_maskable_patched.zip",
        "models/rl_agent_sepsis_masked_final.zip",
        "models/rl_agent_sepsis_final_reproducible.zip",
        "models/debug_stable.zip",
        "models/temp_model.zip",
        "models/rl_agent_sepsis_fast_uf-1_25_2diag_20250828_000125.zip",
        "models/threshold_sepsis.bak",
        "models/sepsis_calibrator_final.bak",
        "models/sepsis_calibrator.joblib",
        "models/roc.svg",
        "models/pr.svg"
    ]
    
    # AKI-related files to delete
    aki_files_to_delete = [
        "models/rl_agent_aki_final.zip",
        "models/rl_agent_aki_single_test_final.zip",
        "models/rl_agent_aki_panels_final.zip",
        "models/rl_aki_ckpt_280000_steps.zip",
        "models/classifier_aki.pth",
        "models/generator_aki.pth",
        "models/gain_scaler_aki.joblib",
        "models/vecnormalize_aki.pkl",
        "models/best_aki/",
        "models/checkpoints_aki/",
        "models/archive/",
        "models/models/",
        "logs_aki_final/",
        "logs_gain_aki/",
        "aki_env/",
        "data/processed/aki/",
        "data/preprocessed/aki_feature_matrix.csv",
        "src/02_preprocess_aki.py",
        "src/02.5_feature_analysis_aki.py",
        "notebooks/03.1_test_augmentation_aki.py"
    ]
    
    # Other unnecessary files
    other_files_to_delete = [
        "test.ipynb",
        "notebooks/atomic-action mapping.ipynb",
        "notebooks/check_SMOTE_CTGAN.ipynb",
        "notebooks/04_proofs.ipynb",
        "baseline/",
        "base paper/"
    ]
    
    # Files to move
    files_to_move = [
        ("00_verify_sepsis_stack.ipynb", "notebooks/00_verify_sepsis_stack.ipynb")
    ]
    
    # Delete Sepsis model duplicates
    print("\n🗑️ Deleting duplicate/outdated Sepsis models...")
    for file_path in sepsis_models_to_delete:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"✅ Deleted: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
        else:
            print(f"⚠️ Not found: {file_path}")
    
    # Delete AKI-related files
    print("\n🗑️ Deleting AKI-related files...")
    for file_path in aki_files_to_delete:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"✅ Deleted: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
        else:
            print(f"⚠️ Not found: {file_path}")
    
    # Delete other unnecessary files
    print("\n🗑️ Deleting other unnecessary files...")
    for file_path in other_files_to_delete:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"✅ Deleted: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
        else:
            print(f"⚠️ Not found: {file_path}")
    
    # Move files to appropriate locations
    print("\n📁 Moving files to appropriate locations...")
    for src, dst in files_to_move:
        if os.path.exists(src):
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                print(f"✅ Moved: {src} → {dst}")
            except Exception as e:
                print(f"❌ Failed to move {src}: {e}")
        else:
            print(f"⚠️ Not found: {src}")
    
    print("\n🎉 Cleanup Complete!")
    print("\n📊 Project Structure After Cleanup:")
    print_project_structure()

def print_project_structure():
    """Print the cleaned project structure"""
    
    print("\n📁 CLEANED SEPSIS PROJECT STRUCTURE:")
    print("=" * 50)
    
    # Core directories
    core_dirs = [
        "configs/",
        "data/processed/sepsis/",
        "data/processed/sepsis/augmented/",
        "models/",
        "logs_sepsis_final/",
        "src/",
        "notebooks/",
        "reports/",
        "docs/"
    ]
    
    for dir_path in core_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
    
    # Essential files
    essential_files = [
        "configs/sepsis_config_accfocus.yaml",
        "models/rl_agent_sepsis_fast_uf-1_25_2diag.zip",
        "models/threshold_sepsis.json",
        "models/generator_sepsis.pth",
        "models/classifier_sepsis.pth",
        "data/processed/sepsis/imputer.joblib",
        "data/processed/sepsis/scaler.joblib",
        "logs_sepsis_final/monitor.csv",
        "logs_sepsis_final/run_metadata.json"
    ]
    
    print("\n📄 ESSENTIAL FILES:")
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")

if __name__ == "__main__":
    cleanup_sepsis_project()
