# 🔍 Fraud Detection System

End-to-end fraud detection system built on the **IEEE-CIS Fraud Detection dataset** (**590K+ transactions**) with a production-style ML pipeline, including:

- **Graph-based fraud ring features** (NetworkX)
- **Threshold-tuned supervised models** (RF / LightGBM / XGBoost)
- **Model comparison + metrics tracking**
- **Streamlit app for interactive fraud scoring**

---

## 🚀 Highlights

| Capability | What it does | Why it matters |
|---|---|---|
| **Graph Features** | Connects cards via shared address relationships and computes graph centrality features | Helps identify coordinated fraud ring behavior |
| **Script-based Pipeline (`src/`)** | Reproducible train/eval workflow via CLI | Better than notebook-only workflows for real projects |
| **Threshold Tuning** | Searches decision threshold using PR curve/F1 | Improves fraud precision-recall tradeoff |
| **Model Registry in Training** | Trains `lr`, `rf`, `lgbm`, `xgb`, or `all` from one script | Easy experimentation without rewriting code |
| **Streamlit App** | UI for model inference and quick demos | Portfolio-ready deployment interface |

---

## 📊 Current Results (latest run)

From your latest successful training/evaluation screenshot:

- **Best model:** `xgb`
- **Best threshold:** `0.6622`
- **Rows used for training:** `120000`

### Evaluation Snapshot

- **Accuracy:** `0.9803`
- **Class 1 (Fraud) Precision:** `0.7491`
- **Class 1 (Fraud) Recall:** `0.6550`
- **Class 1 (Fraud) F1:** `0.6989`
- **Macro F1:** `0.8443`

> Exact model-wise metrics are saved in `models/metrics.json`.

---

## 🗂️ Project Structure

```text
Fraud-Detection/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                    # Kaggle source files (ignored in git)
│   └── clean_train.csv         # engineered dataset (ignored in git)
├── models/
│   ├── best_model_info.json
│   └── metrics.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 2_feature_engg.ipynb
│   └── 3_Modelling.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- **Data/ML:** Pandas, NumPy, scikit-learn
- **Boosting:** LightGBM, XGBoost
- **Graph Features:** NetworkX
- **Deep Learning (notebook experiments):** TensorFlow/Keras (autoencoder)
- **App:** Streamlit
- **Version Control:** Git + GitHub

---

## ⚡ Setup

### 1) Clone & create environment
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

### 2) Prepare data
- Download IEEE-CIS data from Kaggle
- Place raw files under `data/raw/`
- Run preprocessing / feature-engineering pipeline (notebook or scripts) to create `data/clean_train.csv`

---

## 🧪 Training & Evaluation (Script-first workflow)

### Train all models and auto-select best by PR-AUC
```bash
python -m src.train --model all --max_rows 120000
```

### Train one model only
```bash
python -m src.train --model xgb --max_rows 120000
python -m src.train --model lgbm --max_rows 120000
python -m src.train --model rf --max_rows 80000
python -m src.train --model lr --max_rows 120000
```

### Evaluate using saved model + tuned threshold
```bash
python -m src.evaluate
```

---

## 💾 Artifacts

Training saves:

- `models/best_model.joblib` *(ignored in git)*
- `models/metrics.json`
- `models/best_model_info.json`

`best_model_info.json` includes the tuned threshold used in evaluation/app inference.

---

## 🖥️ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📌 Recommended Workflow

- Use **notebooks** for exploration/analysis.
- Use **`src/` scripts as source of truth** for reproducible training/evaluation.
- Keep notebook outputs cleared before commit to avoid repo bloat.

---

## ✅ Roadmap

- [x] EDA and fraud behavior analysis
- [x] Feature engineering (including graph features)
- [x] Reproducible training/evaluation pipeline (`src/`)
- [x] Threshold tuning + metrics tracking
- [x] Streamlit demo app
- [ ] FastAPI inference API
- [ ] Dockerization + deployment
- [ ] CI checks and automated model validation

---

## 👤 Author

**Nischay Kumar**  
GitHub: [NischayKumar04](https://github.com/NischayKumar04)