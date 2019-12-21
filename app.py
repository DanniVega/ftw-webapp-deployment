import pandas as pd
import streamlit as st
import joblib
import numpy as np

#webapp title
st.title('Sales Forecasting')

#description
st.write('Demonstrate how to forecast sales based on ad expenditure.')

#table/read data
data = pd.read_csv('data/advertising_regression.csv')

#show
data

#sidebar
st.sidebar.subheader('Advertising Costs')

TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)
radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25)
newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125)

#histogram/barchart
hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]
st.bar_chart(hist_values)

hist_values = np.histogram(data.TV, bins=300, range=(0,50))[0]
st.bar_chart(hist_values)

hist_values = np.histogram(data.newspaper, bins=300, range=(0,250))[0]
st.bar_chart(hist_values)

#load save ML mowdel
saved_model = joblib.load('advertising_model.sav')

predict_sales = saved_model.predict([[TV, radio, newspaper]])[0]

st.write(f"Predicted sales is {predict_sales} dollars")