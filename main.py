import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style

df = pd.read_csv('Bitcoin.csv')
df.head(5)

# x and y dataframes values
x= df[['High','Open','Low','Volume']].values
y= df['Close'].values

# x and y train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Linear regression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})

# Close price graph
plt.figure(figsize=(10,5))
plt.title('BTC Close Price')
plt.xlabel('Days')
plt.ylabel('Close price USD')
plt.plot(df['Close'])
plt.show()

# Close price df
df =df[['Close']]

# X days prediction variable,minus pred_days
pred_days = 30
df['Prediction']=df[['Close']].shift(-pred_days)

# Convert to numpy array, removing the last x rows/days
X=np.array(df.drop(['Prediction'],1))[:-pred_days]
y = np.array(df['Prediction'])[:-pred_days]

# 75% training and 25% testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# Get the last x row or the future data set
x_future =df.drop(['Prediction'],1)[:-pred_days]
x_future=x_future.tail(pred_days)
x_future=np.array(x_future)
x_future

# Models lr prediction
lr = LinearRegression().fit(x_train,y_train)
lr_prediction = lr.predict(x_future)
print(lr_prediction)

# Data visual

predictions = lr_prediction

valid=df[X.shape[0]:]
valid['Prediction']=predictions
plt.figure(figsize=(15,5))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close price USD')
plt.plot(df['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['Orig','Pred']),
plt.show()