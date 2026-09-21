import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


DROP_COLUMNS = ['id', 'CustomerId', 'Surname']
CATEGORICAL_COLUMNS = ['Geography', 'Gender']


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=columns_to_drop)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)


def align_columns(df: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    return df.reindex(columns=reference_columns, fill_value=0)


def preprocess(df: pd.DataFrame, reference_columns: list = None) -> pd.DataFrame:
    df = clean_data(df)
    df = encode_features(df)
    if reference_columns is not None:
        df = align_columns(df, reference_columns)
    return df


app = FastAPI(title='Bank Churn Prediction API')

model = joblib.load('../models/model.pkl')
feature_columns = joblib.load('../models/feature_columns.pkl')


class Customer(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get('/')
def root():
    return {'message': 'Bank Churn Prediction API is running'}


@app.post('/predict')
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    df_processed = preprocess(df, reference_columns=feature_columns)

    probability = model.predict_proba(df_processed)[:, 1][0]
    prediction = int(probability > 0.5)

    return {
        'churn_probability': round(float(probability), 4),
        'will_churn': bool(prediction)
    }