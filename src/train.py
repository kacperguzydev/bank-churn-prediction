import pandas as pd
import lightgbm as lgb
import joblib
import os

from preprocessing import preprocess


BEST_PARAMS = {
    'n_estimators': 274,
    'max_depth': 5,
    'learning_rate': 0.05433062007888886,
    'num_leaves': 28,
    'min_child_samples': 55,
    'subsample': 0.949495553166147,
    'colsample_bytree': 0.6234830776780291,
    'random_state': 42,
    'verbose': -1
}


def main():
    train = pd.read_csv('../data/train.csv')

    X = train.drop(columns=['Exited'])
    y = train['Exited']

    X = preprocess(X)

    model = lgb.LGBMClassifier(**BEST_PARAMS)
    model.fit(X, y)

    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/model.pkl')
    joblib.dump(X.columns.tolist(), '../models/feature_columns.pkl')

    print('Model trained and saved.')
    print('Feature columns:', X.columns.tolist())


if __name__ == '__main__':
    main()