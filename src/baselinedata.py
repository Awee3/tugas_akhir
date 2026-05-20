import pandas as pd
import json
from pathlib import Path

# 1. Daftar 10 fitur penting sesuai rencana TA kamu
features = ['V14', 'V10', 'V12', 'V4', 'V17', 'V3', 'V11', 'V16', 'V2', 'V9']

BASE_DIR = Path(__file__).resolve().parents[1]  # d:\tugas_akhir
input_path = BASE_DIR / "data" / "creditcard_train.csv"
output_path = BASE_DIR / "data" / "static_baseline.json"

if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

# 2. Load dataset training (64% yang menjadi baseline)
df_train = pd.read_csv(input_path)

missing = [c for c in features if c not in df_train.columns]
if missing:
    raise KeyError(f"Missing columns in training data: {missing}")

# 3. Ekstraksi statistik dasar
baseline_data = {}

for col in features:
    n = min(100, len(df_train))
    baseline_data[col] = {
        "mean": float(df_train[col].mean()),
        "std": float(df_train[col].std()),
        "distribution_sample": df_train[col].sample(n=n, random_state=42).tolist()
    }

# 4. Simpan ke format JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(baseline_data, f, indent=4)

print(f"File static_baseline.json berhasil dibuat: {output_path}")