# 🔍 Fraud Detection System (IEEE-CIS)

An end-to-end fraud detection pipeline built on the IEEE-CIS dataset with reproducible preprocessing, graph-based feature engineering, model training, cost-sensitive thresholding, evaluation, and explainability (SHAP).

---

## ✅ Current Status

- Full data used for training: **590,540 rows**
- Split strategy: **time-based (80/20)** to reduce temporal leakage
- Models trained: **Logistic Regression, Random Forest, LightGBM, XGBoost**
- Feature engineering includes:
  - time features
  - amount/velocity features
  - missingness indicators
  - **graph-based fraud ring features (NetworkX centrality)**
- Model ranking metric: **PR-AUC**
- Serving threshold policy: **cost-sensitive** (`FN_cost`, `FP_cost` configurable)
- Explainability: **SHAP top features export**

---

## 🧠 Why Accuracy Is Not the Main Metric

Fraud detection is highly imbalanced (~3.5% fraud).  
A naive model that always predicts “not fraud” can still get high accuracy, so accuracy is misleading.

This project prioritizes:
- **PR-AUC** (primary ranking metric)
- **ROC-AUC**
- **Class-1 Recall / Precision / F1**
- **Cost-sensitive thresholding** for realistic fraud operations

---

## 📊 Cost-Mode Results Summary (All on 590,540 rows, time-based split)

> These are threshold-policy outcomes from your runs.

### 1) Cost mode: `FN=10`, `FP=1`
- Threshold used: **0.5030**
- Confusion Matrix: `[[108371, 5673], [1375, 2689]]`
- Class 1 Precision: **0.3216**
- Class 1 Recall: **0.6617**
- Class 1 F1: **0.4328**
- Accuracy: **0.9403**

### 2) Cost mode: `FN=20`, `FP=1`
- Best model: **rf**
- Threshold used: **0.3910**
- Confusion Matrix: `[[98480, 15564], [1095, 2969]]`
- Class 1 Precision: **0.1602**
- Class 1 Recall: **0.7306**
- Class 1 F1: **0.2628**
- Accuracy: **0.8590**

### 3) Cost mode: `FN=25`, `FP=1` (from `metrics.json`)
- Best model: **xgb**
- Threshold used: **0.2930**
- Confusion Matrix: `[[99070, 14974], [807, 3257]]`
- Class 1 Precision: **0.1787**
- Class 1 Recall: **0.8014**
- Class 1 F1: **0.2922**
- PR-AUC (xgb): **0.5448**
- ROC-AUC (xgb): **0.9145**

> As FN cost increases, recall rises and precision usually drops — expected for fraud-alert systems.

---

## 🏗️ Project Structure

```text
Fraud-Detection/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── clean_train.csv
├── models/
│   ├── best_model.joblib
│   ├── best_model_info.json
│   ├── metrics.json
│   ├── preprocess_artifacts.joblib
│   ├── preprocess_summary.json
│   ├── shap_top_features.csv
│   └── shap_top_features.png
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 2_feature_engg.ipynb
│   └── 3_Modelling.ipynb
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/NischayKumar04/Fraud-Detection.git
cd Fraud-Detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧹 Preprocessing

Build clean training data from raw files:

```bash
python -m src.preprocess
```

Outputs:
- `data/clean_train.csv`
- `models/preprocess_artifacts.joblib`
- `models/preprocess_summary.json`

---

## 🏋️ Training

Train models and choose threshold policy:

```bash
python -m src.train --model all --max_rows 0 --threshold_mode cost --fn_cost 25 --fp_cost 1
```

Key args:
- `--model`: `all | lr | rf | lgbm | xgb`
- `--threshold_mode`: `f1 | cost`
- `--fn_cost`, `--fp_cost`: relative penalty for missed fraud vs false alert

Saved:
- `models/best_model.joblib`
- `models/best_model_info.json`
- `models/metrics.json`

---

## 📈 Evaluation

```bash
python -m src.evaluate
```

Loads best model + saved threshold and prints:
- confusion matrix
- classification report

---

## 🔮 Batch Prediction

```python
import pandas as pd
from src.predict import predict_batch

df = pd.read_csv("data/clean_train.csv").head(100)
pred_df = predict_batch(df)  # uses saved threshold from best_model_info.json
print(pred_df[["fraud_probability", "fraud_prediction"]].head())
```

---

## 🔎 Model Explainability (SHAP)

This repo includes `src/explain.py` for interpretability.

Run:

```bash
python -m src.explain
```

It:
- loads `models/best_model.joblib`
- computes SHAP values on sampled data
- exports top features to:
  - `models/shap_top_features.csv`
  - `models/shap_top_features.png`

---

## 🖥️ Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Supports:
- CSV upload scoring
- probability + prediction output
- threshold adjustment for scenario testing

---

## 💼 Resume-ready Project Summary

Built a fraud detection pipeline on **590K transactions** with **graph-based fraud ring features (NetworkX centrality)**, time-aware validation, and cost-sensitive thresholding to control precision–recall tradeoffs in imbalanced data.

---

## 🚀 Next Improvements

- [ ] Add time-based cross-validation (mean ± std across folds)
- [ ] Add cost-based **model selection** (not only threshold selection)
- [ ] Add FastAPI endpoint for real-time inference
- [ ] Dockerize training + serving
- [ ] Add schema checks and unit tests

---

## 👤 Author

**Nischay Kumar**  
GitHub: [@NischayKumar04](https://github.com/NischayKumar04)