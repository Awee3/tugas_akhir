import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model_test6.pkl")
TARGET_COL  = "Class"
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

print("[1/3] Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL, "Time"])
y = df[TARGET_COL]

# Stratified split: 80% train | 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
)

# ── Scale Amount ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"]  = scaler.transform(X_test[["Amount"]])

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("[2/3] Training XGBoost...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
ratio = round(neg / pos, 1)

model = XGBClassifier(
    n_estimators=200,
    scale_pos_weight=ratio,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    verbosity=0
)
model.fit(X_train, y_train)
print("  Done.")

# ── Save model ────────────────────────────────────────────────────────────────
print("[3/3] Saving model...")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump({"model": model, "scaler": scaler}, MODEL_OUT)
print(f"  Saved: {MODEL_OUT}")