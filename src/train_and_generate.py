"""
train_and_generate.py
─────────────────────
Versi revisi dari skrip training kamu, diperbaiki agar SEMUA artifact konsisten:

  1. models/xgboost_model.pkl   → bundle {model, scaler, top_features}
  2. static_allbaseline.json    → 29 fitur, key huruf kecil, format {features:{...}, prediction_distribution:{...}}
                                   Amount SCALED (sama dengan ruang yang di-log main.py)
  3. simulasi_normal.csv        → Amount MENTAH (meniru input produksi; API yang men-scale)
  4. simulasi_drift.csv         → Amount MENTAH + drift +3σ pada top-3 fitur (V4, V12, V14)

Jalankan di lingkungan data-science (yang punya creditcard.csv + xgboost):
    python train_and_generate.py
Lalu salin output ke fraud_deploy:
    static_allbaseline.json  → fraud_deploy/monitoring_config/
    simulasi_*.csv           → fraud_deploy/simulation/
    xgboost_model.pkl        → commit ke repo tugas_akhir (untuk n8n) atau taruh di fraud_deploy/models/
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── CONFIG ───────────────────────────────────────────────────────────────────
HERE              = os.path.dirname(__file__)
DATA_PATH         = os.path.join(HERE, "..", "data", "creditcard.csv")
MODEL_OUT         = os.path.join(HERE, "..", "models", "xgboost_model.pkl")
BASELINE_OUT      = os.path.join(HERE, "..", "data", "static_allbaseline.json")
SIM_NORMAL_OUT    = os.path.join(HERE, "..", "data", "simulasi_normal.csv")
SIM_DRIFT_OUT     = os.path.join(HERE, "..", "data", "simulasi_drift.csv")
TARGET_COL        = "Class"
RANDOM_SEED       = 42
BASELINE_SAMPLES  = 5000     # subsample agar KS-Test ringan & file tidak raksasa
N_DRIFT_FEATURES  = 3        # top-3 fitur paling penting → di-shift di simulasi_drift
# ─────────────────────────────────────────────────────────────────────────────

print("[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL, "Time"])     # 29 fitur: V1..V28 + Amount
y = df[TARGET_COL]

# Split: 64% train | 16% test | 20% simulation (stratified, seed sama dgn versi lamamu)
X_traintest, X_sim, y_traintest, y_sim = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
)
X_train, X_test, y_train, y_test = train_test_split(
    X_traintest, y_traintest, test_size=0.20, stratify=y_traintest, random_state=RANDOM_SEED
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Sim: {len(X_sim)}")

# Simpan salinan Amount MENTAH untuk simulasi (sebelum scaling)
X_sim_raw = X_sim.copy()

# ── Scale Amount (untuk training & baseline) ───────────────────────────────────
scaler = StandardScaler()
X_train = X_train.copy(); X_test = X_test.copy()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"]  = scaler.transform(X_test[["Amount"]])

# ── Train XGBoost ──────────────────────────────────────────────────────────────
print("\n[2/6] Training XGBoost...")
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
ratio = round(neg / pos, 1)
model = XGBClassifier(
    n_estimators=200, scale_pos_weight=ratio,
    eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0,
)
model.fit(X_train, y_train)
print(f"  Done. scale_pos_weight={ratio}")

# ── Top features (XGBoost importance) ───────────────────────────────────────────
print("\n[3/6] Extracting top features...")
importances = pd.Series(model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).index.tolist()
top3 = top_features[:N_DRIFT_FEATURES]
print(f"  Top-10: {top_features[:10]}")
print(f"  Top-3 (akan di-shift = Tier-1): {top3}")
print(f"  >>> Set di monitoring.py: TIER1_FEATURES = {[c.lower() for c in top3]}")

# ── Generate static_allbaseline.json (29 fitur, Amount SCALED, key huruf kecil) ──
print("\n[4/6] Generating static_allbaseline.json...")
rng = np.random.default_rng(RANDOM_SEED)
features_block = {}
for col in X_train.columns:                       # 29 fitur (Amount sudah scaled)
    vals = X_train[col].to_numpy()
    if len(vals) > BASELINE_SAMPLES:
        idx = rng.choice(len(vals), BASELINE_SAMPLES, replace=False)
        sampled = vals[idx]
    else:
        sampled = vals
    features_block[col.lower()] = {
        "samples": [round(float(v), 6) for v in sampled],
        "stats": {
            "mean":   round(float(np.mean(vals)), 6),
            "std":    round(float(np.std(vals)), 6),
            "min":    round(float(np.min(vals)), 6),
            "max":    round(float(np.max(vals)), 6),
            "median": round(float(np.median(vals)), 6),
            "q1":     round(float(np.percentile(vals, 25)), 6),
            "q3":     round(float(np.percentile(vals, 75)), 6),
        },
    }

baseline_obj = {
    "features": features_block,
    "prediction_distribution": {
        "fraud_rate": round(float(y_train.mean()), 6),
        "n_train":    int(len(y_train)),
    },
    "meta": {
        "amount_space": "scaled",         # penanda eksplisit: Amount di baseline = scaled
        "scaler_mean":  round(float(scaler.mean_[0]), 6),
        "scaler_scale": round(float(scaler.scale_[0]), 6),
        "tier1_features": [c.lower() for c in top3],
    },
}
with open(BASELINE_OUT, "w") as f:
    json.dump(baseline_obj, f, indent=2)
print(f"  Saved: {BASELINE_OUT} ({len(features_block)} fitur)")

# ── Generate simulasi CSV (Amount MENTAH) ────────────────────────────────────────
print("\n[5/6] Generating simulasi CSV (Amount mentah)...")
sim_normal = X_sim_raw.copy()
sim_normal[TARGET_COL] = y_sim.values
sim_normal.to_csv(SIM_NORMAL_OUT, index=False)
print(f"  simulasi_normal.csv: {len(sim_normal)} baris (Amount mentah)")

sim_drift = X_sim_raw.copy()
for col in top3:                                   # shift +3σ (σ dihitung dari X_train)
    shift = 3 * X_train[col].std()
    sim_drift[col] = sim_drift[col] + shift
    print(f"  +3σ shift '{col}': +{shift:.4f}")
sim_drift[TARGET_COL] = y_sim.values
sim_drift.to_csv(SIM_DRIFT_OUT, index=False)
print(f"  simulasi_drift.csv:  {len(sim_drift)} baris (Amount mentah, drift di {top3})")

# ── Save model bundle ─────────────────────────────────────────────────────────
print("\n[6/6] Saving model bundle...")
joblib.dump({"model": model, "scaler": scaler, "top_features": top_features}, MODEL_OUT)
print(f"  Saved: {MODEL_OUT}")
print("\n✅ Selesai. Pastikan TIER1_FEATURES di monitoring.py = top-3 di atas.")
