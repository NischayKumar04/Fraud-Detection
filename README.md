# 🔍 Fraud Detection System

End-to-end fraud detection system built on **590K+ transactions** from the IEEE-CIS dataset, featuring **graph-based fraud ring detection** and **autoencoder anomaly scoring**.

---

## 🚀 What Makes This Project Unique

| Feature | Description | Why It Matters |
|---------|------------|----------------|
| **Graph-Based Features** | NetworkX graph connecting cards via shared addresses | Detects fraud rings that share billing info |
| **Autoencoder Anomaly Score** | TensorFlow autoencoder trained on legitimate transactions | Catches novel fraud patterns the supervised model hasn't seen |
| **Smart Missing Flags** | Binary flags for missing values before imputation | Preserves signal when fraudsters hide information |

---

## 📊 Results

| Model | AUC-ROC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| Logistic Regression (baseline) | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| LightGBM + SMOTE | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| **Final LightGBM + Anomaly** | **TBD** | **TBD** | **TBD** | **TBD** |

> Results will be updated after Phase 3 (Model Training)

---

## 🗂️ Project Structure

```
Fraud-Detection/
├── data/                            ← Dataset (download from Kaggle)
├── notebooks/
│   ├── 01_eda.ipynb                 ← Phase 1: Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb ← Phase 2: Feature Engineering
│   └── 03_modelling.ipynb           ← Phase 3: Model Training (coming soon)
├── src/                             ← Python scripts
├── models/                          ← Saved models
├── app/                             ← FastAPI + Streamlit (coming soon)
├── requirements.txt
├── Dockerfile                       ← (coming soon)
└── README.md
```

---

## 🛠️ Tech Stack

- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Models:** LightGBM, XGBoost, Scikit-learn
- **Unique Features:** NetworkX (graph), TensorFlow (autoencoder)
- **Deployment:** FastAPI, Docker, Streamlit (coming in Phase 4-5)

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/NischayKumar04/Fraud-Detection.git
cd Fraud-Detection

# Create virtual environment
python -m venv .venv --without-pip
.\.venv\Scripts\activate          # Windows
python -m ensurepip --upgrade
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle and place in data/ folder
# Then run notebooks in order: 01_eda → 02_feature_engineering → 03_modelling
```

---

## 📅 Project Phases

- [x] Phase 1: EDA & Data Understanding
- [x] Phase 2: Feature Engineering (Graph + Smart Missing Flags)
- [ ] Phase 3: Model Training (LightGBM + Autoencoder)
- [ ] Phase 4: FastAPI + Docker Deployment
- [ ] Phase 5: Streamlit Dashboard + GitHub Polish

---

## 📄 Key Findings

- **Class Imbalance:** 1:27 ratio (only ~3.6% fraud)
- **Night-time fraud:** Fraud rate significantly higher during hours 0-5
- **Graph features:** `graph_degree` had the **highest correlation** with fraud (0.1045) among all engineered features — proving fraud ring hypothesis

---

## 👤 Author

**Nischay Kumar** — [GitHub](https://github.com/NischayKumar04)