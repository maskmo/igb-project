import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import numpy as np
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import KFold



x_data = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
x_test = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

cat_col_list = []
for col in x_data.columns:
    if x_data[col].dtype not in ['float', 'int']:
        cat_col_list.append(col)
num_col = x_data.columns.drop(cat_col_list)
cat_col = x_data.columns.drop(num_col)
x_data.drop(columns=cat_col)
x_data.drop(columns=["LotFrontage"])
x_data.dropna()

y = x_data.SalePrice



x_train, x_valid, y_train, y_valid = train_test_split(x_data, y, train_size=0.8, test_size=0.2, random_state=0)

lgbm.early_stopping(stopping_rounds=100)

def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 4020, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1)
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgbm.LGBMRegressor(objective="regression", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l1",
            # early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "l1")
            ],  # Add a pruning callback
        )
        preds = model.predict(X_test)
        cv_scores[idx] = model.score(X_test, y_test)

    return np.mean(cv_scores)



ind = pd.Series(range(1,1169))
x_train = x_train.set_index(ind)
y_train = y_train.reset_index()

lgbm.early_stopping(
                    stopping_rounds=100,
                    first_metric_only=False,
                    verbose=True,
                    min_delta=0.00001,
                )

study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X = x_train[num_col], y = y_train['SalePrice'])
study.optimize(func, n_trials=20)



print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")



params_grid = {
                'n_estimators': 10000,
                'learning_rate': 0.014014497502966943,
                'num_leaves': 60,
                'max_depth': 3,
                'min_data_in_leaf': 200,
                'max_bin': 262,
                'lambda_l1': 40,
                'lambda_l2': 15,
                'min_gain_to_split': 0.8214164700347597,
                'bagging_fraction': 0.4,
                'bagging_freq': 1,
                'feature_fraction': 0.7,
}

model = lgbm.LGBMRegressor(objective="regression", **params_grid)
reg = model.fit(
            x_train[num_col],
            y_train.SalePrice,
            eval_metric="l1",
            eval_set=[(x_valid[num_col], y_valid)],
            early_stopping_rounds=100,
        )
pred = reg.predict(x_valid[num_col])
reg.score(x_valid[num_col], y_valid)
print(pred)