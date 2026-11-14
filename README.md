# Credit Card Fraud Detection — ML Development

This repository contains the machine-learning development work for a credit-card fraud detection project. It covers exploratory data analysis, data preparation, model experimentation, training, and generation of artifacts used by the deployment and monitoring system.

It is the **model-development component** of a two-repository project. The deployment counterpart is:

➡️ **[Awee3/fraud_deploy — Fraud Detection MLOps Platform](https://github.com/Awee3/fraud_deploy)**

## How the two repositories work together

The repositories are intentionally separated by responsibility:

| Repository | Responsibility |
|---|---|
| **[tugas_akhir](https://github.com/Awee3/tugas_akhir)** (this repository) | Dataset exploration, preprocessing, model comparison and training, and generation of model, baseline, and simulation artifacts |
| **[fraud_deploy](https://github.com/Awee3/fraud_deploy)** | FastAPI prediction service, SQLite logging, operational and drift monitoring, Docker deployment, traffic simulation, and n8n automation |

The overall workflow is:

\`\`\`text
Credit-card transaction dataset
              |
              v
    EDA, preprocessing, and split
              |
              v
       Model experiments
              |
              v
       Model training
              |
              +------> Model artifact (.pkl)
              |
              +------> Monitoring baseline (.json)
              |
              +------> Normal / drift simulation data (.csv)
                              |
                              v
                 Awee3/fraud_deploy
                              |
                              v
          FastAPI + SQLite + Monitoring + n8n
\`\`\`

For the API, Docker setup, monitoring implementation, automation workflows, and end-to-end deployment instructions, refer to the **[fraud_deploy README](https://github.com/Awee3/fraud_deploy#readme)**.

## Project scope

The development work uses a credit-card transaction fraud classification dataset. The target column is \`Class\`, where the task is to distinguish fraudulent from legitimate transactions.

The deployment-oriented training pipeline in \`src/train_and_generate.py\` uses the 29-feature API contract:

- \`V1\` through \`V28\`
- \`Amount\`

The \`Time\` and \`Class\` columns are excluded from the model input contract. \`Class\` remains the training target and is retained in generated simulation data where applicable.

## Repository structure

\`\`\`text
tugas_akhir/
├── notebooks/
│   ├── eda01.ipynb
│   └── compare_models.py
├── src/
│   ├── splitdata.py
│   ├── train.py
│   ├── train_xgboost.py
│   ├── train_and_generate.py
│   ├── generate_baseline.py
│   ├── baselinedata.py
│   └── simdrift.py
├── models/
│   └── model artifacts and comparison outputs
├── data/                 # Local dataset and generated files; not tracked by Git
├── requirements.txt
└── README.md
\`\`\`

## Main development components

### Exploratory data analysis

The \`notebooks/eda01.ipynb\` notebook contains exploratory analysis of the credit-card transaction dataset.

### Model comparison

The \`notebooks/compare_models.py\` script compares several classification approaches:

- Logistic Regression
- Decision Tree
- Gaussian Naive Bayes
- Random Forest
- XGBoost
- LightGBM

The script evaluates models using AUC-ROC, PR-AUC, F1-score, precision, and recall, and generates comparison outputs such as ROC curves and confusion matrices.

### Model training and artifact generation

The deployment-oriented script is \`src/train_and_generate.py\`. It:

1. Loads the local credit-card dataset.
2. Creates stratified training, testing, and simulation splits.
3. Scales the \`Amount\` feature using a fitted StandardScaler.
4. Trains an XGBoost classifier with class-imbalance weighting.
5. Extracts feature importance information.
6. Generates a monitoring baseline for the 29 input features.
7. Generates normal and feature-drift simulation datasets.
8. Saves the model together with its scaler and feature metadata.

The generated model bundle is designed to be consumed by the API in **[fraud_deploy](https://github.com/Awee3/fraud_deploy)**.

## Run the deployment-oriented training pipeline

### Requirements

- Python 3.11 or a compatible Python version
- The credit-card transaction dataset used by the scripts
- Python packages: NumPy, pandas, scikit-learn, XGBoost, and joblib

Create and activate a virtual environment (Windows PowerShell):

\`\`\`powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn xgboost joblib
\`\`\`

### Prepare the dataset

The dataset is not included in this repository. Place your local dataset at:

\`\`\`text
data/creditcard.csv
\`\`\`

Create the local data directory if it does not exist:

\`\`\`powershell
New-Item -ItemType Directory -Force data
\`\`\`

### Train and generate artifacts

Run from the repository root:

\`\`\`powershell
python src/train_and_generate.py
\`\`\`

The script generates:

| Artifact | Output path | Purpose |
|---|---|---|
| XGBoost model bundle | \`models/xgboost_model.pkl\` | Model, fitted scaler, and feature metadata for the API |
| Monitoring baseline | \`data/static_allbaseline.json\` | Reference feature distributions and baseline fraud-rate metadata |
| Normal simulation data | \`data/simulasi_normal.csv\` | Normal transaction traffic for testing |
| Drift simulation data | \`data/simulasi_drift.csv\` | Synthetic feature-shift traffic for monitoring tests |

The exact outputs depend on the dataset and training code version. Re-run the script when you intentionally change the training pipeline or its inputs.

## Transfer artifacts to the deployment repository

After generating and validating the artifacts, copy the relevant files into a local clone of **[fraud_deploy](https://github.com/Awee3/fraud_deploy)**:

| From this repository | Destination in \`fraud_deploy\` |
|---|---|
| \`models/xgboost_model.pkl\` | \`models/xgboost_model.pkl\` |
| \`data/static_allbaseline.json\` | \`monitoring_config/static_allbaseline.json\` |
| \`data/simulasi_normal.csv\` and \`data/simulasi_drift.csv\` | \`simulation/\` |

Keep the model, scaler, feature contract, and monitoring baseline compatible. In particular, the API expects the 29 features listed above, and the baseline must describe the same feature representation used by the deployed API.

The deployment repository documents how to run the API, execute simulations, monitor drift, and configure n8n-based automation:

➡️ **[Open fraud_deploy](https://github.com/Awee3/fraud_deploy)**

## Notes

- The dataset and generated CSV files are excluded from Git by the repository's \`.gitignore\`. Obtain the dataset separately and keep local/generated data out of commits unless there is a deliberate reason to publish it.
- The repository contains model experiment artifacts. Not every saved model is necessarily the model currently used by the deployment service.
- For deployment, use an artifact that has been checked against the API's feature contract and the matching monitoring baseline.
