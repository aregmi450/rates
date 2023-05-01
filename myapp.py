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

if st.button('Predict Rate:'):
    price = predict

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
st.title("NPR - USD Exchange Rate")
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
# Use model to make predictions
y_pred = lm.predict(xtest)

# Printout relevant metrics
print("Model Coefficients:", lm.coef_)
print("Mean Absolute Error:", mean_absolute_error(ytest, y_pred))
print("Coefficient of Determination:", r2_score(ytest, y_pred))
# Using LSTM Model to Predict
x = npr[["Open", "High", "Low", "Volume"]]
y = npr["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

# st.dataframe(npr)
# st. table(npr[["Date", "Open", "Close"]])


# xtrain, xtest, ytrain, ytest = train_test_split(
#     x, y, test_size=0.2, random_state=42)
# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))
# model.summary()
# # Training the model
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(xtrain, ytrain, batch_size=1, epochs=50)
# npr.tail(1)
# # features = [Open, High, Low, Adj Close, Volume]
# features = np.array([[127.867599, 127.867599, 127.867599, 0]])
# model.predict(features)
