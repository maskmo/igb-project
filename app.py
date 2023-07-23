
# import necessary libraries for importing, running and displaying model predictions with user inputs
import shap
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import tensorflow_probability as tfp
tfd = tfp.distributions

import pickle
from pickle import dump
from pickle import load

# Necessary for user input data
scaler = MinMaxScaler()

np.random.seed(42)
#set online display settings
st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)


st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames, Iowa</h4>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #1B9E91;'>House price predictions using random forest regression, with displays of shap weights to assess importance of quantitative variables.</h5>", unsafe_allow_html=True)

#create input list of user data for qualitative display and quanititative input
name_list = [
    'MSSubClass',
    'LotFrontage',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'FstFlrSF',
    'SndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageYrBlt',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    'SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold'
    ]

name_list_train = [
    'MSSubClass',
    'LotFrontage',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageYrBlt',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold'
    ]

#import training data
data_input = pd.read_csv('./house-prices/train.csv')

#defining input list of values
data = data_input[name_list_train].values

scaler.fit(data)
#description for user comprehension
description_list = [
    'What is the building class?',
    'How many linear feet of street connected to the property?',
    'What is the lot size in square feet?',
    'What is the Overall material and finish quality?',
    'What is the overall condition of the house from 1-10?',
    'In which year was the Original construction date?',
    'In which year was it remodelled?',
    'What is the area of veneer and masonry in square feet?',
    'What is the type 1 finished square feet?',
    'What is the type 2 finished square feet?',
    'What is the Unfinished square feet of basement area?',
    'What is the Total square feet of basement area?',
    'What is the First Floor square feet?',
    'What is the Second floor square feet?',
    'What is the low quality finished square feet?',
    'What is the Above grade (ground) living area square feet?',
    'What is the number of basement full baths?',
    'What is the number of basement half baths?',
    'What is the number of full bathrooms?',
    'What is the number of Half baths?',
    'What is the number of bedrooms above ground?',
    'What is the number of kitchens above grade?',
    'What is the number of  Total rooms above grade (does not include bathrooms)?',
    'What is the number of fireplaces?',
    'What is the year the garage build?',
    'What is the garage capacity in car sizes?',
    'What is the size of garage in square feet?',
    'What is the area of the wood deck in square feet?',
    'What is the open porch area in square feet?',
    'What is the enclosed porch area in square feet?',
    'What is the three season porch area in square feet?',
    'What is the screen porch area in square feet?',
    'What is the pool area in square feet?',
    'What is the approximate dollar value of unique features?',
    'In which month was it sold?',
    'In which year was it sold?'
 ]

#ranges of inputs available to user

min_list = [20, 21.0, 1470, 1, 1, 1879, 1950, 0.0, 0.0, 0.0, 0.0, 0.0, 407, 0, 0, 407, 0.0, 0.0, 0, 0, 0, 0, 3, 0, 1895.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 1, 2006]

max_list = [190, 200.0, 56600, 10, 10, 2010, 2010, 1290.0, 4010.0, 1526.0, 2140.0, 5095.0, 5095, 1862, 1064, 5095, 3.0, 2.0, 4, 2, 6, 2, 15, 4, 2207.0, 5.0, 1488.0, 1424, 742, 1012, 360, 576, 800, 17000, 12, 2010]

count = 0

#creating UI for input data
with st.sidebar:

    for i in range(len(name_list)):

            

        variable_name = name_list[i]
        globals()[variable_name] = st.slider(description_list[i] ,min_value=int(min_list[i]), max_value =int(max_list[i]),step=1)
      
    st.write("[Kaggle Link to Data Set](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)")

    

#creating object populated by user inputs
data_df_obj = {

'MSSubClass': [MSSubClass],
'LotFrontage': [LotFrontage],
'LotArea': [LotArea],
'OverallQual': [OverallQual],
'OverallCond': [OverallCond],
'YearBuilt': [YearBuilt],
'YearRemodAdd': [YearRemodAdd],
'MasVnrArea': [MasVnrArea],
'BsmtFinSF1': [BsmtFinSF1],
'BsmtFinSF2': [BsmtFinSF2],
'BsmtUnfSF': [BsmtUnfSF],
'TotalBsmtSF': [TotalBsmtSF],
'1stFlrSF': [FstFlrSF],
'2ndFlrSF': [SndFlrSF],
'LowQualFinSF': [LowQualFinSF],
'GrLivArea': [GrLivArea],
'BsmtFullBath': [BsmtFullBath],
'BsmtHalfBath': [BsmtHalfBath],
'FullBath': [FullBath],
'HalfBath': [HalfBath],
'BedroomAbvGr': [BedroomAbvGr],
'KitchenAbvGr': [KitchenAbvGr],
'TotRmsAbvGrd': [TotRmsAbvGrd],
'Fireplaces': [Fireplaces],
'GarageYrBlt': [GarageYrBlt],
'GarageCars': [GarageCars],
'GarageArea': [GarageArea],
'WoodDeckSF': [WoodDeckSF],
'OpenPorchSF': [OpenPorchSF],
'EnclosedPorch': [EnclosedPorch],
'3SsnPorch': [SsnPorch],
'ScreenPorch': [ScreenPorch],
'PoolArea': [PoolArea],
'MiscVal': [MiscVal],
'MoSold': [MoSold],
'YrSold': [YrSold] 


}

#converting the user input dictionairy to dataframe
data_df = pd.DataFrame.from_dict(data_df_obj)

#importing random forest regressor from pickle
with open('model_pickle.pkl', 'rb') as f:
    model = pickle.load(f)



#generating the prediction based on user input
pred = model.predict(data_df)




#creating UI instances
col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('Calculate range of house price')

#creating shap values and metrics to be displayed
explainer = shap.TreeExplainer(model)
sv = explainer(data)

exp = Explanation(sv.values,
                  sv.base_values[0][0],
                  data=data_df.values,
                  feature_names=name_list_train)

shap_values = explainer.shap_values(data)


#manually creating an st component that can write a shap graph on the app page
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


#creating user input button when finished inputting data to generate predictions and shap visuals
if center_button:

    import time


    with st.spinner('Calculating....'):
        time.sleep(2)



    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>The price range of your house is between:</h5>", unsafe_allow_html=True)


    col1, col2 = st.columns([3, 3])

    lower_number = "{:,.2f}".format(int(pred-17496.299733333333))
    higher_number = "{:,.2f}".format(int(pred+17496.299733333333))

    col1, col2, col3= st.columns(3)
    
    

    with col1:
        st.write("")

    with col2:
        st.subheader("USD "+ str(lower_number))
        st.subheader("       AND ")

        st.subheader(" USD "+str(higher_number))
        st_shap(
            shap.force_plot(
                explainer.expected_value,
                shap_values[0,:],
                data_df[name_list_train].iloc[0,:],
                matplotlib=False
                )
            )

    with col3:

        st.write("")



