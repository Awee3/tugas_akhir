import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model_test6.pkl")
TARGET_COL  = "Class"
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

print("[1/4] Loading data...")
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

print(f"  Train: {len(X_train)} rows | Fraud: {y_train.sum()}")
print(f"  Test : {len(X_test)} rows | Fraud: {y_test.sum()}")
print(f"  Sim  : {len(X_sim)} rows | Fraud: {y_sim.sum()}")

# ── Scale Amount ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"]  = scaler.transform(X_test[["Amount"]])
X_sim["Amount"]   = scaler.transform(X_sim[["Amount"]])

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\n[2/4] Training XGBoost...")
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

# ── Evaluate model ───────────────────────────────────────────────────────────
print("\n[3/4] Evaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"  Accuracy : {acc:.6f}")
print(f"  Precision: {prec:.6f}")
print(f"  Recall   : {rec:.6f}")
print(f"  F1-Score : {f1:.6f}")
print(f"  ROC-AUC  : {roc:.6f}")
print(f"  PR-AUC   : {pr_auc:.6f}")
print("  Confusion Matrix:")
print(cm)
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ── Save model ────────────────────────────────────────────────────────────────
print("\n[4/4] Saving model...")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump({"model": model, "scaler": scaler}, MODEL_OUT)
print(f"  Saved: {MODEL_OUT}")