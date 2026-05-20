import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_OUT      = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")
BASELINE_OUT   = os.path.join(os.path.dirname(__file__), "..", "data", "static_baseline.json")
SIM_NORMAL_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "simulasi_normal.csv")
SIM_DRIFT_OUT  = os.path.join(os.path.dirname(__file__), "..", "data", "simulasi_drift.csv")
TARGET_COL     = "Class"
RANDOM_SEED    = 42
TOP_N_FEATURES = 10
# ─────────────────────────────────────────────────────────────────────────────

print("[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL, "Time"])
y = df[TARGET_COL]

# Stratified split: 64% train | 16% test | 20% simulation
X_traintest, X_sim, y_traintest, y_sim = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
)
X_train, X_test, y_train, y_test = train_test_split(
    X_traintest, y_traintest, test_size=0.20, stratify=y_traintest, random_state=RANDOM_SEED
)

print(f"  Train : {len(X_train)} rows | Fraud: {y_train.sum()}")
print(f"  Test  : {len(X_test)} rows  | Fraud: {y_test.sum()}")
print(f"  Sim   : {len(X_sim)} rows   | Fraud: {y_sim.sum()}")

# ── Scale Amount ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"]  = scaler.transform(X_test[["Amount"]])
X_sim["Amount"]   = scaler.transform(X_sim[["Amount"]])

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\n[2/6] Training XGBoost...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
ratio = round(neg / pos, 1)
print(f"  scale_pos_weight = {ratio}")

model = XGBClassifier(
    n_estimators=200,
    scale_pos_weight=ratio,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    verbosity=0
)
model.fit(X_train, y_train)
print("  Done.")

# ── Feature Importance — Top 10 ───────────────────────────────────────────────
print("\n[3/6] Extracting Top 10 features...")
importances = pd.Series(model.feature_importances_, index=X_train.columns)
top10 = importances.sort_values(ascending=False).head(TOP_N_FEATURES).index.tolist()
print(f"  Top 10 features: {top10}")

# ── Generate static_baseline.json ─────────────────────────────────────────────
print("\n[4/6] Generating static_baseline.json...")
baseline = {}
for col in top10:
    values = X_train[col].tolist()
    baseline[col] = {
        "mean":  round(float(np.mean(values)), 6),
        "std":   round(float(np.std(values)), 6),
        "min":   round(float(np.min(values)), 6),
        "max":   round(float(np.max(values)), 6),
        "p25":   round(float(np.percentile(values, 25)), 6),
        "p50":   round(float(np.percentile(values, 50)), 6),
        "p75":   round(float(np.percentile(values, 75)), 6),
        "values": values  # full distribution for KS-test in n8n
    }

with open(BASELINE_OUT, "w") as f:
    json.dump({"top_features": top10, "baseline": baseline}, f, indent=2)
print(f"  Saved: {BASELINE_OUT}")

# ── Save simulation CSVs ───────────────────────────────────────────────────────
print("\n[5/6] Saving simulation sets...")
sim_df = X_sim.copy()
sim_df[TARGET_COL] = y_sim.values

# Normal simulation — untouched
sim_normal = sim_df.copy()
sim_normal.to_csv(SIM_NORMAL_OUT, index=False)
print(f"  simulasi_normal.csv saved: {len(sim_normal)} rows")

# Drift simulation — inject 3σ mean shift on top critical features
DRIFT_FEATURES = top10[:3]  # shift the top 3 most important features
sim_drift = sim_df.copy()
for col in DRIFT_FEATURES:
    shift_amount = 3 * X_train[col].std()
    sim_drift[col] = sim_drift[col] + shift_amount
    print(f"  Injected 3σ shift on '{col}': +{shift_amount:.4f}")

sim_drift.to_csv(SIM_DRIFT_OUT, index=False)
print(f"  simulasi_drift.csv saved: {len(sim_drift)} rows")

# ── Save model ────────────────────────────────────────────────────────────────
print("\n[6/6] Saving model...")
joblib.dump({"model": model, "scaler": scaler, "top_features": top10}, MODEL_OUT)
print(f"  Saved: {MODEL_OUT}")

print("\n✅ All done! Files generated:")
print(f"   Model    → {MODEL_OUT}")
print(f"   Baseline → {BASELINE_OUT}")
print(f"   Sim norm → {SIM_NORMAL_OUT}")
print(f"   Sim drift→ {SIM_DRIFT_OUT}")