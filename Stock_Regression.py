

import pandas as pd

import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 4, 6)

df = web.DataReader("AAPL", 'yahoo', start, end)
print(df.tail())

df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0



forecast_col = 'Adj Close'


df.fillna(method='ffill', inplace=True)
forecast_out = 30
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.tail())



df = df.drop(labels='Adj Close', axis=1)
import numpy as np

X = np.array(df.drop(['label'], 1))

from sklearn import preprocessing

X = preprocessing.scale(X)
print(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
print(X)

df.dropna(inplace=True)
y = np.array(df['label'])
print(y)

from sklearn import model_selection
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
confidencelr = lr.score(X_test, y_test)

print("Linear Regression ",confidencelr)

forecast_set_lr = lr.predict(X_lately)
print("Linear Regression forecast ",forecast_set_lr)

from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)
confidencereg = reg.score(X_test, y_test)
print("Ridge Regression ",confidencereg)
forecast_set_reg = reg.predict(X_lately)
print("Ridge Regression forecast",forecast_set_reg)

from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
confidencelasso_reg = lasso_reg.score(X_test, y_test)
print("Lasso Regression",confidencelasso_reg)
forecast_set_lasso_reg = lasso_reg.predict(X_lately)
print("Lasso Regression Forecast",forecast_set_lasso_reg)



df['Forecast'] = np.nan
last_date = df.iloc[-1].name
print(last_date)

last_unix = last_date.timestamp()
print(last_unix)

one_day = 86400

next_unix = last_unix + one_day
print(next_unix)


for i in forecast_set_lr:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



for i in forecast_set_reg:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]



import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


for i in forecast_set_lasso_reg:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

style.use('ggplot')

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





