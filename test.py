import joblib, pandas as pd
b = joblib.load("models/xgboost_model.pkl")
m = b["model"]
imp = pd.Series(m.feature_importances_, index=m.feature_names_in_).sort_values(ascending=False)
print(imp.head(10).round(4))
print("Porsi kumulatif top-3:", round(imp.head(3).sum() / imp.sum(), 3))