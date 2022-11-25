import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv('DiamondsPrices.csv')

df = df.dropna(axis = 0)
df_trans = df.drop(['x', 'y', 'z'], axis=1)
df_trans.to_csv("DiamondsPrices.csv")
df_trans = pd.get_dummies(df_trans)

X = df_trans.drop(['price'], axis = 1)

y = df_trans['price']

s = StandardScaler()
X = s.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)
lin = LinearRegression()
lin = lin.fit(X_train,y_train)
y_pred1 = lin.predict(X_test)

rf = RandomForestRegressor(n_estimators = 50)
rf.fit(X_train,y_train)
y_pred3 = y_pred = rf.predict(X_test)

model_eval = pd.DataFrame(index = ['LM','RF'], columns = ['RMSE'])

ypred_null = y_train.mean() #benchmark using the record average

model_eval.loc['LM','RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred1))
model_eval.loc['RF','RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred3))

model_eval.to_csv("metrics.csv")
RF_model_path = "RF_model.pkl"
LM_model_path = "LM_model.pkl"
with open(RF_model_path, 'wb') as f:
    pickle.dump(rf, f)

with open(LM_model_path, 'wb') as f:
    pickle.dump(lin, f)