# 🏦 Credit Risk Classification using Logistic Regression

This project is a **credit risk modeling** solution that predicts whether a loan applicant is likely to default or not, based on their demographic, financial, and credit history data. The solution uses **Logistic Regression**, with hyperparameter tuning via **Optuna**, SMOTE for balancing, and a Streamlit UI for live demo.

---

## 📌 Problem Statement

Financial institutions need to assess the creditworthiness of applicants to reduce default rates. The goal is to build a binary classification model that predicts:

- `1` → Loan Default (High Risk)
- `0` → Non-Default (Low Risk)

---

## 🧠 Key Features

| Feature Category | Description |
|------------------|-------------|
| Personal Details | Age, Gender, Marital Status |
| Loan Info | Loan Amount, Term, EMI |
| Credit History | Credit Score, Bureau Data |
| Target Variable | Default (Yes/No) |

---

## 📂 Dataset Used

The project uses 3 merged datasets:

- `customers.csv` — Personal details of loan applicants.
- `loans.csv` — Loan information per customer.
- `bureau_data.csv` — Credit history from external bureaus.

---

## ⚙️ Modeling Pipeline

### ✅ 1. Exploratory Data Analysis (EDA)
- Merged datasets on `customer_id`
- Handled missing values and outliers
- Engineered new features: e.g. `loan_to_income`, `age_bucket`

### ✅ 2. Feature Engineering
- Label encoding for categorical variables
- Created new ratio-based and bucketed features
- Optional: WOE and IV calculations (planned)

### ✅ 3. Preprocessing
- Data split into train/test
- Used **SMOTE** to handle class imbalance

### ✅ 4. Model Training
- Base model: `LogisticRegression`
- Tuning with **Optuna** on F1-score
- 3-fold cross-validation
- Final model trained on best parameters

### ✅ 5. Evaluation
- Metrics: F1-score, precision, recall, ROC-AUC
- Classification report and confusion matrix
- Feature importance analyzed

---

## 🔍 Hyperparameter Tuning (Optuna)

Optuna was used to tune:
- `C`: Regularization strength
- `solver`: Algorithm (lbfgs, saga, liblinear, etc.)
- `tol`: Optimization tolerance
- `class_weight`: Balanced vs. unbalanced

---

## 📈 Results

| Metric | Value |
|--------|-------|
| F1 Score (CV) | ~0.80 |
| Test F1 Score | 0.78–0.82 |
| ROC AUC | 0.84+ |
| Best Solver | `liblinear` (varies per run) |

---

## 🖥️ Streamlit Demo

To run the app locally:

```bash
git clone https://github.com/shekhus/ml-project-credit-risk-model.git
cd credit-risk-model
streamlit run main.py
