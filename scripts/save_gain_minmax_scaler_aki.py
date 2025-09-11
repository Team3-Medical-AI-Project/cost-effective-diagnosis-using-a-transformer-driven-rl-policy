# scripts/save_gain_minmax_scaler_aki.py
import pandas as pd, joblib
from sklearn.preprocessing import MinMaxScaler

X = pd.read_csv("data/processed/aki/train_X.csv")
sc = MinMaxScaler().fit(X.values)
joblib.dump(sc, "models/gain_scaler_aki.joblib")
print("✅ saved models/gain_scaler_aki.joblib")
