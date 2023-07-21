import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


x_data = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
x_test = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

x_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = x_data.SalePrice
x_data.drop(['SalePrice'], axis=1, inplace=True)

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y, train_size=0.8, test_size=0.2, random_state=0)

categorical_cols = [cname for cname in x_train.columns if
                    x_train[cname].dtype == "object"]

numerical_cols = [cname for cname in x_train.columns if 
                x_train[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = x_train[my_cols].copy()
X_valid = x_valid[my_cols].copy()
X_test = x_test[my_cols].copy()


numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor()

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

clf.fit(X_train, y_train)
# 
# preds = clf.predict(X_valid)
# 
# print('MAE:', mean_absolute_error(y_valid, preds))
print(X_test)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.force_plot(explainer.expected_value, shap_values[0,:])