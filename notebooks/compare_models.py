import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "comparison")
TARGET_COL  = "Class"
TEST_SIZE   = 0.20   # consistent with your 64/16/20 split → test uses the 16% slice
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────────────────


def load_and_split(path, test_size, seed):
    """Load CSV and do a stratified train/test split (no SMOTE — clean evaluation)."""
    from sklearn.model_selection import train_test_split

    print(f"[1/5] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape} | Fraud rate: {df[TARGET_COL].mean()*100:.4f}%")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Drop 'Time' column — usually not useful after PCA features are present
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])

    # Scale 'Amount' — only non-PCA feature left that needs it
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    print(f"      Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"      Train fraud count: {y_train.sum()} | Test fraud count: {y_test.sum()}")
    return X_train, X_test, y_train, y_test


def get_models(pos_weight):
    """Return dict of candidate models. pos_weight handles class imbalance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_SEED, max_depth=10
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_SEED,
            verbosity=0, use_label_encoder=False
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_SEED, verbose=-1
        ),
    }


def evaluate(model, X_test, y_test):
    """Return a dict of all metrics for one model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 4),
        "PR-AUC":    round(average_precision_score(y_test, y_prob), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
    }, y_pred, y_prob


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train all models and collect results."""
    results     = {}
    predictions = {}
    probas      = {}

    print("\n[3/5] Training & evaluating models...")
    for name, model in models.items():
        print(f"      ▶ {name}...", end=" ", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        metrics, y_pred, y_prob = evaluate(model, X_test, y_test)
        metrics["Train time (s)"] = round(elapsed, 2)

        results[name]     = metrics
        predictions[name] = y_pred
        probas[name]      = y_prob
        print(f"AUC={metrics['AUC-ROC']} | F1={metrics['F1']} | done in {elapsed:.1f}s")

    return results, predictions, probas, models


def print_results_table(results):
    """Print a clean comparison table to the console."""
    df = pd.DataFrame(results).T
    df = df.sort_values("AUC-ROC", ascending=False)

    print("\n" + "="*75)
    print("  MODEL COMPARISON RESULTS")
    print("="*75)
    print(df.to_string())
    print("="*75)
    print("\n★  Best AUC-ROC :", df["AUC-ROC"].idxmax(), f"({df['AUC-ROC'].max()})")
    print("★  Best F1      :", df["F1"].idxmax(),       f"({df['F1'].max()})")
    print("★  Best Recall  :", df["Recall"].idxmax(),   f"({df['Recall'].max()})")
    print("★  Best PR-AUC  :", df["PR-AUC"].idxmax(),   f"({df['PR-AUC'].max()})")
    return df


def plot_roc_curves(models_dict, probas, y_test, save_dir):
    """Plot all ROC curves on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

    for (name, model), color in zip(models_dict.items(), colors):
        RocCurveDisplay.from_predictions(
            y_test, probas[name], name=name, ax=ax, color=color
        )

    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"      Saved: {path}")


def plot_metric_bars(results_df, save_dir):
    """Bar chart comparing AUC, F1, Precision, Recall per model."""
    metrics = ["AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]
    df_plot = results_df[metrics].sort_values("AUC-ROC", ascending=False)

    x     = np.arange(len(df_plot))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df_plot[metric], width, label=metric, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90
            )

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_plot.index, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "metric_bars.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"      Saved: {path}")


def plot_confusion_matrices(models_dict, predictions, y_test, save_dir):
    """6-panel confusion matrix grid."""
    n   = len(models_dict)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (name, _) in zip(axes, models_dict.items()):
        cm = confusion_matrix(y_test, predictions[name])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"      Saved: {path}")


def save_csv(results_df, save_dir):
    path = os.path.join(save_dir, "model_comparison.csv")
    results_df.to_csv(path)
    print(f"      Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH, TEST_SIZE, RANDOM_SEED)

    # 2. Compute class imbalance ratio for XGBoost scale_pos_weight
    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    ratio = round(neg / pos, 1)
    print(f"\n[2/5] Class ratio (neg/pos): {ratio}  → used as XGBoost scale_pos_weight")

    # 3. Build models
    models_dict = get_models(ratio)

    # 4. Train & evaluate
    results, predictions, probas, fitted_models = train_and_evaluate(
        models_dict, X_train, X_test, y_train, y_test
    )

    # 5. Print table
    print("\n[4/5] Printing results table...")
    results_df = print_results_table(results)

    # 6. Plots
    print("\n[5/5] Generating plots...")
    plot_roc_curves(fitted_models, probas, y_test, OUTPUT_DIR)
    plot_metric_bars(results_df, OUTPUT_DIR)
    plot_confusion_matrices(fitted_models, predictions, y_test, OUTPUT_DIR)
    save_csv(results_df, OUTPUT_DIR)

    print(f"\n✅ All done! Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()