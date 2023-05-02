# Importing required modules
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('USD-NPR RATE CONVERTER')

st.write('This project contains dataset having exchange rates of USD in Nepal which will predict the rate of conversion.')

st.image('dollarbills.jpg')


sns.set()
plt.style.use('seaborn-whitegrid')
# Reading the dataset and looking the information on the dataset
npr = pd.read_csv('Conversion.csv')
npr.head()
# Finding out the information about dataset data
npr.info()
# converting the date column to datetime format
npr['Date'] = pd.to_datetime(npr['Date'])
print(
    f'This dataset contains exchange rates of USD in Nepal from {npr.Date.min()} {npr.Date.max()}')
print(f'Total Days = {(npr.Date.max() - npr.Date.min()).days} days')
npr.describe()

# Plot showing rate of change of exchange rate
st.title("NPR - USD Exchange Rate Graph")
graph = plt.figure(figsize=(10, 4))
plt.xlabel('Date')
plt.ylabel("Close")
plt.plot(npr['Date'], npr['Close'])
st.pyplot(graph)

# Using Linear Regression Model to Predict
# splitting data into train and test sets

x = np.array(npr.index).reshape(-1, 1)
y = npr['Close']
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42)
# Feature Scaling
scaler = StandardScaler().fit(xtrain)
lm = LinearRegression()
lm.fit(xtrain, ytrain)
price = lm.predict(xtest)

# Use model to make predictions
st.write('The regression model predicts the rate of USD to be following using Linear Regression Model')
st.write(price)

# Printout relevant metrics
print("Model Coefficients:", lm.coef_)
print("Mean Absolute Error:", mean_absolute_error(ytest, price))
print("Coefficient of Determination:", r2_score(ytest, price))
