# generate_baseline.py
import pandas as pd
import numpy as np
import json
import os

TRAINING_DATA_PATH = "data/creditcard_train.csv"       # Sesuaikan path
OUTPUT_PATH        = "data/static_baseline.json"

# Semua fitur yang akan dimonitor
ALL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Jumlah sampel yang disimpan per fitur untuk KS-Test
# Trade-off: lebih banyak = lebih akurat, tapi file lebih besar
# 2000 samples sudah cukup untuk KS-Test yang reliable
N_SAMPLES = 2000


def generate_baseline(df: pd.DataFrame) -> dict:
    baseline = {
        "metadata": {
            "created_at":         pd.Timestamp.now().isoformat(),
            "training_set_size":  len(df),
            "features_monitored": len(ALL_FEATURES),
            "n_samples_per_feature": N_SAMPLES
        },
        "features":               {},
        "prediction_distribution": {}
    }

    for feature in ALL_FEATURES:
        col = df[feature]

        # Simpan N_SAMPLES random samples untuk KS-Test
        # (tidak perlu simpan semua data training)
        samples = col.sample(
            n=min(N_SAMPLES, len(col)),
            random_state=42
        ).tolist()

        baseline["features"][feature.lower()] = {
            "samples": samples,
            # Statistik deskriptif untuk inspeksi manual
            "stats": {
                "mean":   round(float(col.mean()), 6),
                "std":    round(float(col.std()),  6),
                "min":    round(float(col.min()),  6),
                "max":    round(float(col.max()),  6),
                "median": round(float(col.median()), 6),
                "q1":     round(float(col.quantile(0.25)), 6),
                "q3":     round(float(col.quantile(0.75)), 6),
            }
        }

    # Distribusi prediksi baseline (dari data training)
    # Gunakan kolom Class jika ada
    if "Class" in df.columns:
        fraud_rate = float(df["Class"].mean())
        baseline["prediction_distribution"] = {
            "fraud_rate":      round(fraud_rate, 6),
            "legitimate_rate": round(1 - fraud_rate, 6),
            "total_samples":   len(df)
        }

    return baseline


if __name__ == "__main__":
    print(f"[Baseline] Loading training data from: {TRAINING_DATA_PATH}")
    df = pd.read_csv(TRAINING_DATA_PATH)
    
    print(f"[Baseline] Generating baseline for {len(ALL_FEATURES)} features...")
    baseline = generate_baseline(df)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    
    print(f"[Baseline] ✅ Saved to: {OUTPUT_PATH}")
    print(f"[Baseline] Features covered: {len(baseline['features'])}")
    print(f"[Baseline] Fraud rate in training: {baseline['prediction_distribution']['fraud_rate']:.4%}")