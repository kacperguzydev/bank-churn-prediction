# Bank Customer Churn Prediction — End-to-End ML System

Predicting which bank customers are likely to leave, using a full pipeline from raw data to a deployed, interactive prediction system.

---

## Problem

Banks lose revenue when customers close their accounts. Retention campaigns are expensive to run blindly — this project predicts *which* customers are at risk of churning, so the bank can act before they leave rather than after.

Dataset: [Kaggle Playground Series S4E1 — Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1) (165,034 customers, 14 features).

## Result

| Metric | Score |
|---|---|
| ROC-AUC (local test set) | 0.8900 |
| ROC-AUC (Kaggle private leaderboard) | 0.8887 |
| F1-score (churn class) | 0.63 |

**0.8887 places in the top 1% of this competition's leaderboard.**

## Approach

1. **EDA** — identified the strongest churn signals: number of products (non-linear relationship — customers with 3–4 products churn far more than those with 2), age, account activity, and geography (Germany churns 2x more than France/Spain).
2. **Preprocessing** — one-hot encoding, stratified train/test split to preserve the 79/21 class imbalance.
3. **Handling class imbalance** — compared plain Logistic Regression, SMOTE oversampling, and `scale_pos_weight`. Found that a stronger model (LightGBM) handled the imbalance better than forcing balance through SMOTE.
4. **Model comparison** — Logistic Regression → XGBoost → LightGBM, selected on ROC-AUC and F1 trade-offs, not accuracy (which is misleading on imbalanced data).
5. **Hyperparameter tuning** — Optuna, 30 trials, 5-fold cross-validation.
6. **Explainability** — SHAP confirmed the model learned the same patterns found in EDA (product count and age as top drivers), validating that the model reflects real business signal rather than noise.
7. **Deployment** — model served via a FastAPI REST endpoint, with a Streamlit dashboard for non-technical users to test predictions live.

## Architecture

```
Data → EDA → Preprocessing → Model training (LightGBM + Optuna) → SHAP
                                        ↓
                                  models/model.pkl
                                        ↓
                        FastAPI (/predict endpoint)
                                        ↓
                        Streamlit dashboard (user-facing)
```

## Tech stack

- **Data & modeling:** pandas, numpy, scikit-learn, XGBoost, LightGBM, Optuna, imbalanced-learn, SHAP
- **Serving:** FastAPI, uvicorn
- **Interface:** Streamlit
- **Deployment:** Render

## Project structure

```
bank-churn-prediction/
├── data/                   # train/test data (Kaggle)
├── notebooks/              # EDA and model development
├── src/                    # reusable preprocessing + training scripts
├── api/                    # FastAPI serving endpoint
├── dashboard/              # Streamlit UI
├── models/                 # trained model + preprocessing artifacts
└── requirements.txt
```

## Running locally

```bash
git clone https://github.com/kacperguzydev/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt

# Train the model
cd src
python train.py

# Start the API
cd ../api
uvicorn main:app --reload

# In a separate terminal, start the dashboard
cd ../dashboard
streamlit run app.py
```

## API usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 45,
    "Tenure": 3,
    "Balance": 0,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000
  }'
```

Response:
```json
{
  "churn_probability": 0.9573,
  "will_churn": true
}
```

## Author

Kacper Guzy — [GitHub](https://github.com/kacperguzydev)
