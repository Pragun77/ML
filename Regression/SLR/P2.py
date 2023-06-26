#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing A Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[ : , :-1].values #stores the independant values of dataset into x
y = dataset.iloc[ : , -1].values #stores the dependant values into y

#Splitting the dataset into Training Set and Test Set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size= 0.2 , random_state= 1)

#Training the Simple Linear Regression Model on the training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #builds the model
regressor.fit(x_train, y_train) #trains the simple linear regression model on training sets

#Predicting the Test set results
y_pred = regressor.predict(x_test) #predicted salaries of the test set

#Visualising The Training Set Results
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising The Test Set Results
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()