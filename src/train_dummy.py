import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dummy_model2.pkl")

def train_dummy(sample_frac=0.05):
    print("1. Memuat dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File tidak ditemukan di {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Ambil sample kecil agar cepat
    if 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("2. Membagi data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("3. Melatih DummyClassifier (baseline cepat)...")
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(X_train, y_train)

    print("4. Evaluasi Singkat:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"5. Menyimpan model ke {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Selesai! Model dummy siap untuk testing deployment.")

if __name__ == "__main__":
    train_dummy()