import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]   # d:\tugas_akhir
input_path = BASE_DIR / "data" / "simulasi_produksi.csv"
output_path = BASE_DIR / "data" / "sim_drift.csv"

if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

df_sim = pd.read_csv(input_path)

if "V14" not in df_sim.columns:
    raise KeyError(f"Column 'V14' not found. Available columns: {list(df_sim.columns)}")

sigma_v14 = df_sim["V14"].std()
df_sim["V14"] = df_sim["V14"] + (3 * sigma_v14)

df_sim.to_csv(output_path, index=False)
print(f"File created: {output_path}")