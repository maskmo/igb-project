import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



x_data = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
x_test = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

y = x_data.SalePrice
x_data.drop(['SalePrice'], axis=1, inplace=True)

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y, train_size=0.8, test_size=0.2, random_state=0)



cat_col_list = []
for col in x_train.columns:
    if x_train[col].dtype not in ['float', 'int']:
        cat_col_list.append(col)
num_col = x_train.columns.drop(cat_col_list)

cat_col = x_train.columns.drop(num_col)



numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_col),
        ('cat', categorical_transformer, cat_col)
    ])

model = RandomForestRegressor()

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

clf.fit(x_train,y_train)



preds_valid = clf.predict(x_valid)

print('MAE:', mean_absolute_error(y_valid, preds_valid))



preds_test = clf.predict(x_test)