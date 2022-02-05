import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('Bitcoin.csv')
df.shape

# BTC Close price graphic

plt.figure(figsize=(15,5))
plt.title('BTC Close Price')
plt.xlabel('Days')
plt.ylabel('Close price in USD')
plt.plot(df['Close'])
plt.show()

# Close dataframe
df = df[['Close']]

# Days prediction
prediction_days = 30

# Create a new column(Prediction), - prediction_days
df['Prediction'] = df['Close'].shift(-prediction_days)
df.tail(2)

#Create X and Y array
X = np.array(df.drop(['Prediction'],1))[:-prediction_days]
print(X)
y = np.array(df['Prediction'])[:-prediction_days]

# Training and testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#Linear Regression
tree = DecisionTreeRegressor().fit(x_train,y_train)
lr = LinearRegression().fit(x_train,y_train)

x_future = df.drop(['Prediction'],1)[:-prediction_days]
x_future = x_future.tail(prediction_days)
x_future = np.array(x_future)
x_future

# Model tree prediction

tree_pred = tree.predict(x_future)
print(tree_pred)
print()

# Linear regression prediction

lr_pred = lr.predict(x_future)
print(lr_pred)

# Prediction graphic

predictions = tree_pred

valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close price USD')
plt.plot(df['Close'])
plt.plot(valid[['Close','Prediction']])
plt.show


