# 🔍 Fraud Detection System (IEEE-CIS)

An end-to-end fraud detection pipeline using the IEEE-CIS dataset (**590,540 transactions**), with reproducible preprocessing, model training, threshold tuning, evaluation, and prediction APIs.

---

## 🚀 Project Highlights

- ✅ Full dataset training (`rows_used = 590,540`)
- ✅ Time-based split to reduce leakage risk
- ✅ Multiple models: Logistic Regression, Random Forest, LightGBM, XGBoost
- ✅ Threshold search per model (not fixed 0.5 only)
- ✅ Artifact-based deployment (`best_model.joblib`, `best_model_info.json`, `metrics.json`)
- ✅ Streamlit-ready prediction workflow

---

## 📊 Latest Results (from `models/metrics.json`)

**Best model:** `xgb`  
**Best threshold:** `0.7676`  
**Best PR-AUC:** `0.5512`  
**Split type:** `time_based`

### Tuned metrics by model

| Model | PR-AUC | ROC-AUC | F1 (tuned) | Precision (tuned) | Recall (tuned) | Best Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1785 | 0.8317 | 0.3204 | 0.2854 | 0.3652 | 0.8566 |
| Random Forest | 0.4586 | 0.8779 | 0.4581 | 0.4913 | 0.4291 | 0.6335 |
| LightGBM | 0.5395 | 0.9117 | 0.5295 | 0.6183 | 0.4631 | 0.8192 |
| **XGBoost (Best)** | **0.5512** | **0.9181** | **0.5366** | **0.6159** | **0.4754** | **0.7676** |

> Note: At default threshold `0.5`, recall is higher for some models, but tuned thresholds optimize F1/precision-recall tradeoff.

---

## 🧱 Repository Structure

```text
Fraud-Detection/
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── clean_train.csv
├── models/
│   ├── best_model.joblib
│   ├── best_model_info.json
│   ├── metrics.json
│   ├── preprocess_artifacts.joblib
│   └── preprocess_summary.json
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 2_feature_engg.ipynb
│   └── 3_Modelling.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/NischayKumar04/Fraud-Detection.git
cd Fraud-Detection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧹 Preprocessing

Build clean training data from raw Kaggle files:

```bash
python -m src.preprocess
```

This creates:
- `data/clean_train.csv`
- `models/preprocess_artifacts.joblib`
- `models/preprocess_summary.json`

---

## 🏋️ Training

Train all models and automatically select best one by PR-AUC:

```bash
python -m src.train --model all --max_rows 0
```

Arguments:
- `--model`: `all | lr | rf | lgbm | xgb`
- `--max_rows 0`: use full dataset (debug cap if >0)

Outputs:
- `models/best_model.joblib`
- `models/best_model_info.json`
- `models/metrics.json`

---

## 📈 Evaluation

```bash
python -m src.evaluate
```

- Loads best model + tuned threshold from artifacts
- Prints confusion matrix + classification report

---

## 🔮 Prediction

Example usage in Python:

```python
import pandas as pd
from src.predict import predict_batch

df = pd.read_csv("data/clean_train.csv").head(100)
out = predict_batch(df)  # uses saved tuned threshold automatically
print(out[["fraud_probability", "fraud_prediction"]].head())
```

---

## 🖥️ Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Supports:
- CSV upload scoring
- sample batch scoring
- threshold slider for business tuning

---

## 📌 Current Notes

- `preprocess_summary.json` currently stores absolute Windows paths.  
  For portability, prefer relative paths in future update.
- Threshold was optimized for F1; if your business needs higher recall, use a lower threshold in serving (e.g., 0.5–0.65) and monitor false positives.

---

## 🛣️ Next Improvements

- [ ] Add graph features from notebook directly into `src/preprocess.py`
- [ ] Add leakage-safe target encoding in `src` pipeline
- [ ] Add cross-validation (time-series folds)
- [ ] Add SHAP explainability report
- [ ] Add `tests/` (smoke + metric sanity + schema validation)
- [ ] Add FastAPI inference endpoint + Dockerized deployment

---

## 👤 Author

**Nischay Kumar**  
GitHub: [@NischayKumar04](https://github.com/NischayKumar04)