


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

import tensorflow_probability as tfp
tfd = tfp.distributions

import pickle
from pickle import dump
from pickle import load

# dump(scaler, open('scaler.pkl', 'wb'))

#scaler = load(open('scaler.pkl', 'rb'))

scaler = MinMaxScaler()



tf.random.set_seed(42)

np.random.seed(42)


st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)



st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames, Iowa</h4>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #1B9E91;'>House price predictions using random forest regression, with displays of shap weights to assess importance of quantitative variables.</h5>", unsafe_allow_html=True)


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

data = pd.read_csv('./house-prices/train.csv')


data = data[name_list_train].values

scaler.fit(data)

description_list = [
    'What is the building class?',
    'What is the Overall material and finish quality?',
    'In which year was the Original construction date?',
    'In which year was it remodelled?',
    'What is the Unfinished square feet of basement area?',
    'What is the Total square feet of basement area?',
    'What is the First Floor square feet?',
    'What is the Second floor square feet?',
    'What is the Above grade (ground) living area square feet?',
    'What is the number of full bathrooms?',
    'What is the number of Half baths?',
    'What is the number of  Total rooms above grade (does not include bathrooms)?',
    'What is the number of fireplaces?',
    'What is the garage capacity in car sizes?',
    'What is the size of garage in square feet?',
    'In which month was it sold?',
    'In which year was it sold?',
    'How many linear feet of street connected to the property?',
    'What is the lot size in square feet?',
    'What is the overall condition of the house from 1-10?',
    'What is the area of veneer and masonry in square feet?',
    'What is the type 1 finished square feet?',
    'What is the type 2 finished square feet?',
    'What is the low quality finished square feet?',
    'What is the number of basement full baths?',
    'What is the number of basement half baths?',
    'What is the number of bedrooms above ground?',
    'What is the number of kitchens above grade?',
    'What is the year the garage build?',
    'What is the area of the wood deck in square feet?',
    'What is the open porch area in square feet?',
    'What is the enclosed porch area in square feet?',
    'What is the three season porch area in square feet?',
    'What is the screen porch area in square feet?',
    'What is the pool area in square feet?',
    'What is the approximate dollar value of unique features?'
 ]



min_list = [20.0,
    1.0,
    1872.0,
    1950.0,
    0.0,
    0.0,
    334.0,
    0.0,
    334.0,
    0.0,
    0.0,
    2.0,
    0.0,
    0.0,
    0.0,
    1.0,
    2000.0,
    20,
    1000,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1900,
    0,
    0,
    0,
    0,
    0,
    0,
    0]

max_list = [190.0,
 10.0,
 2010.0,
 2010.0,
 2336.0,
 6110.0,
 4692.0,
 2065.0,
 5642.0,
 3.0,
 2.0,
 14.0,
 3.0,
 4.0,
 1800.0,
 12.0,
 2020.0,
 350,
 250000,
 10,
 1900,
 2400,
 1200,
 600,
 3,
 3,
 6,
 3,
 2012,
 900,
 600,
 600,
 510,
 500,
 700,
 2500]

count = 0

with st.sidebar:

    for i in range(len(name_list)):

            

        variable_name = name_list[i]
        globals()[variable_name] = st.slider(description_list[i] ,min_value=int(min_list[i]), max_value =int(max_list[i]),step=1)
      
    st.write("[Kaggle Link to Data Set](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)")

    


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

data_df = pd.DataFrame.from_dict(data_df_obj)


with open('model_pickle', 'rb') as f:
    model = pickle.load(f)


# negloglik = lambda y, p_y: -p_y.log_prob(y) # note this

# model1 = tf.keras.models.load_model('model_files/my_keras_model1.h5')

# model1 = tf.keras.models.Sequential(model1.layers[:5])

# data_df = pd.DataFrame.from_dict(data_df)

# data_df_normal = scaler.transform(data_df)

pred = model.predict(data_df)

# model2 = tf.keras.models.load_model('model_files/keras_2.h5',compile=False)

# yhat = model2(latent_var)



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



if center_button:

    import time

    #my_bar = st.progress(0)

    with st.spinner('Calculating....'):
        time.sleep(2)



    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>The price range of your house is between:</h5>", unsafe_allow_html=True)


    col1, col2 = st.columns([3, 3])

    lower_number = "{:,.2f}".format(int(pred-17496.299733333333))
    higher_number = "{:,.2f}".format(int(pred+17496.299733333333))

    col1, col2, col3 = st.columns(3)

    

    with col1:
        st.write("")

    with col2:
        st.subheader("USD "+ str(lower_number))
        st.subheader("       AND ")

        st.subheader(" USD "+str(higher_number))


    with col3:
        st.write("")

    

    

    # import base64

    # # file_ = open("kramer_gif.gif", "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()

    # st.markdown(
    #     f'<center><img src="data:image/gif;base64,{data_url}" alt="cat gif"></center>',
    #     unsafe_allow_html=True,
    # )