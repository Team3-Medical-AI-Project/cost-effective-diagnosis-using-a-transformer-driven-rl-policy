"""
AKI Feature Importance Analysis (v1.1 - Corrected)

v1.1: Fixes a plotting keyword error (color vs. palette).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- 1. Configuration ---
INPUT_FILE = "data/preprocessed/aki_feature_matrix.csv"
TARGET_COLUMN = "kdigo_aki"
ID_COLUMNS = ['subject_id', 'hadm_id', 'stay_id']

# --- 2. Load and Prepare Data ---
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
X = df.drop(columns=[TARGET_COLUMN] + ID_COLUMNS)
y = df[TARGET_COLUMN]

# Use a simple split for this analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data loaded and split.")

# --- 3. Train XGBoost Model ---
print("Training XGBoost model to determine feature importance...")
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Extract and Visualize Feature Importance ---
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(40) # Let's look at the top 40

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 10))

# --- THE FIX ---
# Changed the keyword from 'color' to 'palette'
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax)

ax.set_title('Top 40 Most Important Features for AKI Prediction', fontsize=16)
ax.set_xlabel('XGBoost Feature Importance', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

print("\n--- Analysis Complete ---")
print("The plot above shows the most predictive features for AKI in your dataset.")