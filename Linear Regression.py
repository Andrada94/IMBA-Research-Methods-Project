# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 20:10:22 2021

@author: Andrada Mitea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model


df=pd.read_csv('D:\Desktop\Alchool & drug usage.csv')
df.head()
df.describe()
df.plot(x='Airliner Fatalities',y='Airliner Accidents',style='*',color='r')
plt.xlabel('Airliner Fatalities')
plt.ylabel('Airliner Acctidents')


X = df.iloc[:,0].values
y=df.iloc[:,1].values
X=np.reshape(X,(-1,1))
y=np.reshape(y,(-1,1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#create a linear regression object 
reg=linear_model.LinearRegression()
# Fit the data using the Linear regression model
reg.fit(X_train,y_train)
#predicting the test set results
y_pred=reg.predict(X_test)
print(y_pred)
#Calculating the coefficient
print(reg.coef_)
#Calculating te Intercept
print(reg.intercept_)

#Evaluating the model-> Calculating the R squared value 
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
print(r2_score)
plt.plot(X_test,y_pred)
plt.show()