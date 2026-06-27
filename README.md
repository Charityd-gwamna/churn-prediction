# Customer Churn Prediction Dashboard

**by Charity David**

> *"Customer churn is not a service problem. It is a revenue problem."*

A machine learning dashboard that predicts which telecom customers are likely to cancel their subscription — before they do. Built with Python, Scikit-learn, and Streamlit, deployed live on Streamlit Cloud.

**Live App**: [churn-prediction-charity.streamlit.app](https://churn-prediction-charity.streamlit.app/)  
**Notebook**: `churn_eda.ipynb`  
**Dataset**: [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Business Problem

A US telecommunications company is losing **26.5% of its customers** every period — representing an estimated **$139,000 in monthly recurring revenue at risk**. The business had no systematic way to identify which customers were likely to leave before they cancelled.

This project answers one question:

> **Can we predict which customers will churn so the retention team can intervene before it's too late?**

---

## Key Findings

| Segment | Churn Rate | vs Average |
|---------|-----------|------------|
| Month-to-month contract | 42.7% | +16.2pp |
| Electronic check payment | 45.3% | +18.8pp |
| Fiber optic internet | 41.9% | +15.4pp |
| No online security | 41.8% | +15.3pp |
| Senior citizens | 41.7% | +15.2pp |
| Two-year contract | 2.8% | -23.7pp |

**Top model predictors:** TotalCharges → Tenure → MonthlyCharges → Contract Type → Internet Service

**Surprising finding:** Gender has near-zero predictive power (importance: 0.026). Churn is driven by service experience and pricing, not demographics.

---

## Model

| Detail | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Class imbalance handling | `class_weight='balanced'` |
| Train / Test split | 80% / 20% (stratified) |
| Overall Accuracy | 79% |
| ROC-AUC Score | 0.83 |
| Precision (Churn) | 64% |
| Recall (Churn) | 49% |

---

## Dashboard Pages

| Page | What It Shows |
|------|--------------|
| Overview | Revenue at risk, churn rate, executive summary, key risk factors |
| EDA | 4-tab exploratory analysis: churn overview, tenure & charges, services, demographics |
| Predict | Input any customer profile — instant churn probability with top reasons |
| Feature Importance | Ranked chart of what drives churn in the model |
| Recommendations | 7 data-driven business actions to reduce churn |

---

## Business Recommendations

1. Convert month-to-month customers to annual contracts — cuts churn from 42.7% to 2.8%
2. Investigate fiber optic satisfaction — highest-paying segment churning the most
3. Bundle security and support services — reduces two compounding risk factors
4. Incentivise auto-pay adoption — electronic check users churn at 45.3% vs 15.2% for auto-pay
5. Create a senior citizen retention programme — 41.7% churn in this segment
6. Focus on early tenure onboarding — most churn happens in months 1-2
7. Deploy model proactively — score all customers monthly, flag above 50% for retention outreach

---

## Tech Stack

```
Python • Streamlit • Scikit-learn • Pandas • NumPy • Matplotlib • Seaborn • Pickle
```

---

## Run Locally

```bash
git clone https://github.com/charityd-gwamna/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
churn-prediction/
│
├── app.py                                  # Streamlit dashboard
├── churn_eda.ipynb                         # EDA and model training notebook
├── churn_model.pkl                         # Trained Random Forest model
├── model_features.pkl                      # Feature column names
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── requirements.txt                        # Dependencies
└── README.md
```

---

## Author

**Charity David** — Data Scientist  
[GitHub](https://github.com/charityd-gwamna)
