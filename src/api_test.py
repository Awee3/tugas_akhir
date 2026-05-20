import requests, pandas as pd

# Load one row from simulasi_normal.csv to test
df = pd.read_csv("tugas_akhir/data/simulasi_normal.csv")
row = df.drop(columns=["Class"]).iloc[0].to_dict()

response = requests.post("http://localhost:5000/predict", json=row)
print(response.json())
# Expected: {"prediction": 0, "label": "LEGIT", "probability": 0.00xxxxx}

health = requests.get("http://localhost:5000/health")
print(health.json())
# Expected: {"status": "ok", "model": "xgboost", "top_features": [...]}