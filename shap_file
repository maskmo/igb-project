import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from shap import TreeExplainer, Explanation
from shap.plots import waterfall

x_data = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
x_test = pd.read_csv('/Users/max/Downloads/house-prices-advanced-regression-techniques/test.csv', index_col='Id')


cat_col_list = []
for col in x_data.columns:
    if x_data[col].dtype not in ['float', 'int']:
        cat_col_list.append(col)
num_col = x_data.columns.drop(cat_col_list)
cat_col = x_data.columns.drop(num_col)

x_data = x_data[num_col].dropna()

y = x_data.SalePrice
x_data.drop(['SalePrice'], axis=1, inplace=True)
num_col = num_col.drop('SalePrice')

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y, train_size=0.8, test_size=0.2, random_state=0)



model = RandomForestRegressor()



model.fit(x_train, y_train)


preds = model.predict(x_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test[num_col])

shap.force_plot(
    explainer.expected_value,
    shap_values[0,:],
    x_test[num_col].iloc[0,:],
    matplotlib=True
)


shap.summary_plot(shap_values,
                  features = num_col)


sv = explainer(x_test[num_col])

exp = Explanation(sv.values,
                  sv.base_values[0][0],
                  data=x_test[num_col].values,
                  feature_names=num_col)

shap.plots.waterfall(exp[0])