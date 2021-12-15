# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 20:10:22 2021

@author: Andrada Mitea    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model

#Read and display the dataset from the .csv file.
df=pd.read_csv('D:\Desktop\Aviation accidents_dataset1.csv')
df.head()
df.describe()

#Visualize the data using plot function (and label the axes) 
df.plot(x='Airliner Fatalities',y='Airliner Accidents',style='*',color='r')
plt.xlabel('Airliner Accident')
plt.ylabel('Airliner Fatalities ')

#Define X, Y dependent and independent variables from our csv and reshape the arrays without changing the data
X = df.iloc[:,2].values 
y=df.iloc[:,1].values
X=np.reshape(X,(-1,1))
y=np.reshape(y,(-1,1))

#Split the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#create a linear regression object 
reg=linear_model.LinearRegression()

# Fit the data using the Linear regression model
reg.fit(X_train,y_train)

#Calculating the coefficient
b1=reg.coef_
print('The b1 coefficient is',b1)


#Calculating te Intercept
b0=reg.intercept_
print('The b0 coefficient is',b0)

#Evaluating the model-> Calculating the R squared value 
r_sq=reg.score(X_train,y_train)
print('Coefficint of determination:',r_sq)

#Predicting the test set results
y_pred=reg.predict(X_train)
print(y_pred)

#Plot the final results
plt.plot(X_train,y_pred)
plt.show()


#Read and display the dataset from the .csv file.
df=pd.read_csv("D:\Desktop\Aviation accidents_dataset2.csv")
df.head()

#Erase the target array from our dataset and set the testing number of target as well as the training number of samples.
inputs=df.drop('aircraft_damage',axis='columns')
target=df['aircraft_damage']
train_target=target.head(8000)
test_target=target.tail(2000)

# Encode the input
from sklearn.preprocessing import LabelEncoder 
le_event_date=LabelEncoder()
le_country=LabelEncoder()
le_amateur_built=LabelEncoder()
le_engine_type=LabelEncoder()
le_airport_code=LabelEncoder()
le_aircraft_category=LabelEncoder()
le_registration_number=LabelEncoder()
le_make=LabelEncoder()
le_model=LabelEncoder()
le_purpose_of_flight=LabelEncoder()
le_weather_condition=LabelEncoder()

inputs['event_date_n']=le_event_date.fit_transform(inputs['event_date'].astype(str))
inputs['country_n']=le_country.fit_transform(inputs['country'].astype(str))
inputs['amateur_built_n']=le_amateur_built.fit_transform(inputs['amateur_built'].astype(str))
inputs['engine_type_n']=le_engine_type.fit_transform(inputs['engine_type'].astype(str))
inputs['airport_code_n']=le_airport_code.fit_transform(inputs['airport_code'].astype(str))
inputs['aircraft_category_n']=le_aircraft_category.fit_transform(inputs['aircraft_category'].astype(str))
inputs['registration_number_n']=le_registration_number.fit_transform(inputs['registration_number'].astype(str))
inputs['make_n']=le_make.fit_transform(inputs['make'].astype(str))
inputs['model_n']=le_model.fit_transform(inputs['model'].astype(str))
inputs['purpose_of_flight_n']=le_purpose_of_flight.fit_transform(inputs['purpose_of_flight'].astype(str))
inputs['weather_condition_n']=le_weather_condition.fit_transform(inputs['weather_condition'].astype(str))

#Create a new dataframe and drop the existing labels for simplicity
inputs_n=inputs.drop(['event_date','country','airport_code','aircraft_category','registration_number','make','model','amateur_built','number_of_engines', 'engine_type','purpose_of_flight','weather_condition'],axis='columns')

#Choose your data which will need to be trained/modeles into the classification 
train_inputs=inputs_n.head(8000)
test_inputs=inputs_n.tail(2000)

#Import the model of Decision Tree classifier and fit our data (train_inputs and train_target) into the model
from sklearn import tree 
model=tree.DecisionTreeClassifier()
model.fit(train_inputs,train_target)

#Predict the values for our testing data
model.score(train_inputs,train_target)
predicted_values=model.predict(test_inputs)


#Check the accuracy score to see what is the accray of our model
from sklearn.metrics import accuracy_score
print('The accuracy score is',accuracy_score(predicted_values,test_target))

# Check our model by plotting the confusion matrix and see how it performed
import seaborn as sn 
confusion_matrix = pd.crosstab(test_target, predicted_values, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show()



