import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset asli
# Pastikan file creditcard.csv berada di folder yang sesuai [cite: 307]
df = pd.read_csv('data/creditcard.csv')

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Split Pertama: Memisahkan data untuk Simulasi Produksi (20%)
# Data ini 'disembunyikan' dari model dan hanya digunakan untuk simulasi di n8n [cite: 258]
X_dev, X_sim, y_dev, y_sim = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Split Kedua: Membagi sisa data menjadi Training (80% dari dev) dan Testing (20% dari dev)
# Training digunakan untuk membuat baseline statis dan melatih model [cite: 228, 305]
X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
)

# 4. Simpan ke file CSV terpisah
# Gabungkan kembali fitur dan target sebelum disimpan
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
sim_production_data = pd.concat([X_sim, y_sim], axis=1)

train_data.to_csv('data/creditcard_train.csv', index=False)
test_data.to_csv('data/creditcard_test.csv', index=False)
sim_production_data.to_csv('data/simulasi_produksi.csv', index=False)

# 5. Verifikasi Hasil
print("Dataset berhasil di-split:")
print(f"- Training Set: {len(train_data)} baris (Fraud: {train_data['Class'].sum()})")
print(f"- Testing Set : {len(test_data)} baris (Fraud: {test_data['Class'].sum()})")
print(f"- Simulation Set: {len(sim_production_data)} baris (Fraud: {sim_production_data['Class'].sum()})")