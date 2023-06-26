#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing A Dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[ : , :-1].values #stores the independant values of dataset into x
y = dataset.iloc[ : , -1].values #stores the dependant values into y

#Encoding Categorial data
#ENCODING THE INDEPENDANT VARIABLE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #convert our categories into numbers 

ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [3])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Splitting the dataset into Training Set and Test Set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size= 0.2 , random_state= 1)

#Training the multiple linear regression model on the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting The Test Results
y_pred  = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Evaluating the model performance
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)