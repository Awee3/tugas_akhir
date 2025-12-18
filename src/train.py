import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model_v2.pkl')

def train():
    print("1. Memuat dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File tidak ditemukan di {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Persiapan Data
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split Data
    print("2. Membagi data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training
    print("3. Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluasi Singkat
    print("4. Evaluasi Model:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Simpan Model
    print(f"5. Menyimpan model ke {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Selesai! Model siap digunakan.")

if __name__ == "__main__":
    train()